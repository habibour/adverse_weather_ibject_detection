import os
import torch
import torch.nn as nn
from tqdm import tqdm
from utils.utils import get_lr

def fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, gen, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    loss        = 0
    Dehazy_loss = 0
    Det_loss    = 0
    criterion = nn.MSELoss()

    # Dynamic Weight Averaging (DWA) task weights, recomputed once per epoch
    # from the previous two epochs' average losses (equal weights until then,
    # or always equal/fixed-lambda-compatible if config.USE_DWA is False).
    w_det, w_dehaze = yolo_loss.get_task_weights()

    if local_rank == 0:
        print('Start Train')
        print('Task weights this epoch -> detection: %.3f, dehaze: %.3f' % (w_det, w_dehaze))
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
        model_train.train()

    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        images, targets, clean = batch[0], batch[1], batch[2]
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                targets = targets.cuda(local_rank)
                clean = clean.cuda(local_rank)
                hazy_and_clear = torch.cat([images, clean], dim = 0).cuda()
        optimizer.zero_grad()

        if not fp16:
            outputs         = model_train(hazy_and_clear)
            detect_outputs = [outputs[0],outputs[1],outputs[2]]
            loss_detection      = yolo_loss(detect_outputs, targets, images)
            loss_dehazy     = criterion(outputs[3], clean)
            loss_value      = w_det * loss_detection + 0.1 * w_dehaze * loss_dehazy
            loss_value.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                outputs         = model_train(images)
                loss_value      = yolo_loss(outputs, targets, images)
            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()
        if ema:
            ema.update(model_train)
        Dehazy_loss += loss_dehazy.item()
        Det_loss    += loss_detection.item()
        loss        += loss_value.item()
        if local_rank == 0:
            pbar.set_postfix(**{'loss'  : loss / (iteration + 1),
                                'loss_detection'  : Det_loss / (iteration + 1),
                                'Dehazy_loss': Dehazy_loss / (iteration + 1),
                                'lr'    : get_lr(optimizer)})
            pbar.update(1)

    # Feed this epoch's raw (unweighted) task losses back into the DWA
    # history so next epoch's weights can be computed. Harmless no-op when
    # config.USE_DWA is False.
    yolo_loss.update_task_loss_history(Det_loss / epoch_step, Dehazy_loss / epoch_step)

    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()

    if local_rank == 0:
        pbar.close()
        loss_history.append_loss(epoch + 1, loss / epoch_step)
        eval_callback.on_epoch_end(epoch + 1, model_train_eval)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f' % (loss / epoch_step))
        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f.pth" % (epoch + 1, loss / epoch_step)))

        # Best-checkpoint selection: prefer validation mAP (what we actually
        # care about) once it's available, since train loss stops being a
        # reliable proxy for accuracy when DWA is reweighting the loss
        # formulation epoch to epoch. Falls back to train loss only on the
        # epochs before the first mAP evaluation has run.
        real_maps       = eval_callback.maps[1:]
        just_evaluated  = len(real_maps) > 0 and eval_callback.epoches[-1] == epoch + 1
        if just_evaluated and eval_callback.maps[-1] >= max(real_maps):
            # NOTE: eval_callback.maps stores mAP as a 0-1 fraction, not a
            # percentage -- multiply by 100 so this print isn't misleading
            # (previously printed e.g. "0.74%" for an actual 74.4% mAP).
            print('Save best model to best_epoch_weights.pth (val mAP=%.2f%%)' % (eval_callback.maps[-1] * 100))
            torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))
        elif not real_maps and loss / epoch_step <= min(loss_history.losses):
            print('Save best model to best_epoch_weights.pth (train loss, no mAP eval yet)')
            torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))
        torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))

        # --- Periodic evaluation on the REAL VOC-FOG-test / RTTS-test sets ---
        # eval_callback above only ever evaluates VOC-FOG's own *validation*
        # split. Every REAL_EVAL_PERIOD epochs (and at the final epoch) we
        # additionally run the exact same subprocess-isolated eval_one.py used
        # for the final baseline/control/proposed comparison against the real
        # held-out test sets, and diff the result against the baseline
        # checkpoint's numbers -- so regressions are visible during training,
        # not just discovered at the end. Config fields are written by the
        # notebook's write_config(); if they're absent (e.g. a plain baseline/
        # control/proposed eval run, not a training run) this is a no-op.
        # Wrapped in try/except so a failure here (OOM, bad path, etc.) can
        # never take down an in-progress training run.
        try:
            from config import (REAL_EVAL_PERIOD, VOCFOG_TEST_IMAGES_DIR, VOCFOG_TEST_ANN_DIR,
                                 RTTS_IMAGES_DIR, RTTS_ANN_DIR, RTTS_IDS_FILE,
                                 classes_path as _eval_classes_path, anchors_path as _eval_anchors_path,
                                 BASELINE_VOCFOG_MAP, BASELINE_RTTS_MAP)
            _real_eval_enabled = True
        except ImportError:
            _real_eval_enabled = False

        if _real_eval_enabled and ((epoch + 1) % REAL_EVAL_PERIOD == 0 or epoch + 1 == Epoch):
            import subprocess, json as _json
            try:
                tmp_ckpt = os.path.join(save_dir, 'tmp_real_eval.pth')
                torch.save(save_state_dict, tmp_ckpt)
                results = {}
                for dataname, images_dir, ann_dir, ext, ids_file in [
                    ('VOC-FOG-test', VOCFOG_TEST_IMAGES_DIR, VOCFOG_TEST_ANN_DIR, '.jpg', None),
                    ('RTTS', RTTS_IMAGES_DIR, RTTS_ANN_DIR, '.png', RTTS_IDS_FILE),
                ]:
                    out_json = os.path.join(save_dir, f'real_eval_ep{epoch + 1}_{dataname}.json')
                    cmd = ['python', 'eval_one.py', '--dataname', dataname, '--model_path', tmp_ckpt,
                           '--images_dir', images_dir, '--ann_dir', ann_dir, '--image_ext', ext,
                           '--classes_path', _eval_classes_path, '--anchors_path', _eval_anchors_path,
                           '--out_json', out_json]
                    if ids_file:
                        cmd += ['--ids_file', ids_file]
                    print(f'[real-eval] epoch {epoch + 1}: running {dataname} on the real test set...')
                    ret = subprocess.run(cmd, capture_output=True, text=True)
                    if ret.returncode != 0:
                        print(f'[real-eval] {dataname} FAILED at epoch {epoch + 1} (continuing training):')
                        print(ret.stdout[-1500:])
                        print(ret.stderr[-1500:])
                        continue
                    with open(out_json) as f:
                        results[dataname] = _json.load(f)

                if 'VOC-FOG-test' in results:
                    m = results['VOC-FOG-test']['mAP']
                    if BASELINE_VOCFOG_MAP is not None:
                        print('[real-eval] epoch %d | VOC-FOG-test mAP=%.2f%% | baseline=%.2f%% | delta=%+.2f%%'
                              % (epoch + 1, m, BASELINE_VOCFOG_MAP, m - BASELINE_VOCFOG_MAP))
                    else:
                        print('[real-eval] epoch %d | VOC-FOG-test mAP=%.2f%% | (no baseline configured)'
                              % (epoch + 1, m))
                if 'RTTS' in results:
                    m = results['RTTS']['mAP']
                    if BASELINE_RTTS_MAP is not None:
                        print('[real-eval] epoch %d | RTTS mAP=%.2f%% | baseline=%.2f%% | delta=%+.2f%%'
                              % (epoch + 1, m, BASELINE_RTTS_MAP, m - BASELINE_RTTS_MAP))
                    else:
                        print('[real-eval] epoch %d | RTTS mAP=%.2f%% | (no baseline configured)'
                              % (epoch + 1, m))

                history_path = os.path.join(save_dir, 'real_eval_history.json')
                history = []
                if os.path.exists(history_path):
                    with open(history_path) as f:
                        history = _json.load(f)
                history.append({'epoch': epoch + 1, **{k: v.get('mAP') for k, v in results.items()}})
                with open(history_path, 'w') as f:
                    _json.dump(history, f, indent=2)
                if os.path.exists(tmp_ckpt):
                    os.remove(tmp_ckpt)
            except Exception as e:
                print(f'[real-eval] WARNING: periodic real-test eval failed at epoch {epoch + 1}: {e!r} '
                      f'-- training continues.')
