import logging
import numpy as np
import cv2
import os
import torch

from stft_core.structures.boxlist_ops import boxlist_iou
from stft_core.structures.bounding_box import BoxList

from .cvcvideo_eval import cvcvideo_detection_eval, cvcvideo_localization_center_eval


def vid_cvcvideo_evaluation(dataset, predictions, output_folder, visulize, vis_thr, iou_threshold, **_):
    logger = logging.getLogger("stft_core.inference")
    logger.info(" performing cvcvideo evaluation.")


    pred_boxlists = []
    gt_boxlists = []
    filename_lists = []
    for image_id, prediction in enumerate(predictions):
        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        
        # 예측 바운딩 박스 리사이징 및 클리핑
        if prediction.size[0] != image_width or prediction.size[1] != image_height:
            prediction = prediction.resize((image_width, image_height))
        prediction = prediction.clip_to_image(remove_empty=True)
        pred_boxlists.append(prediction)

        # GT 바운딩 박스 처리
        gt_boxlist = dataset.get_groundtruth(image_id)
        if gt_boxlist.size[0] != image_width or gt_boxlist.size[1] != image_height:
            gt_boxlist = gt_boxlist.resize((image_width, image_height))
        gt_boxlist = gt_boxlist.clip_to_image(remove_empty=True)
        gt_boxlists.append(gt_boxlist)

        filename_lists.append(dataset.get_img_name(image_id))

    if output_folder:
        torch.save(pred_boxlists, os.path.join(output_folder, "pred_boxlists.pth"))
        torch.save(gt_boxlists, os.path.join(output_folder, "gt_boxlists.pth"))
        torch.save(filename_lists, os.path.join(output_folder, "filename_lists.pth"))
    
    score_thrs = np.arange(0.5, 0.6, 0.1)
    #score_thrs = np.arange(0.2, 0.3, 0.1)   # prediction threshold range
    
    logger.info(" Polyp Detection Task:")
    det_evals_dict = {}
    det_evals, det_tp, det_fp, det_tn, det_fn = cvcvideo_detection_eval(pred_boxlists, gt_boxlists, score_thrs, iou_threshold)
    det_metrics = ['Precision', 'Recall', 'Accuracy', 'Sepcificity', 'F1_score', 'F2_score']
    for i in range(score_thrs.shape[0]):
        pt_string = '\nscore_thr:{:.2f}'.format(score_thrs[i])
        for j in range(len(det_metrics)):
            pt_string += '  {}: {:.4f} '.format(det_metrics[j], det_evals[j][i])
            each_name = '{}/score_thr:{:.2f}'.format(det_metrics[j], score_thrs[i])
            each_iterm = det_evals[j][i]
            det_evals_dict[each_name] = each_iterm
        logger.info(pt_string)

    if output_folder and visulize:
        #저장할 폴더 생성
        for folder in ['TP', 'FP', 'TN', 'FN']:
            os.makedirs(os.path.join(output_folder, folder), exist_ok=True)
            
        for image_id, (gt_boxlist, pred_boxlist) in enumerate(zip(gt_boxlists, pred_boxlists)):
            img, target, filename = dataset.get_visualization(image_id)
            save_line = filename+' '

            # 실제 이미지 크기 확인
            img_height, img_width = img.shape[:2]

            # GT 바운딩 박스 그리기
            gt_bbox = gt_boxlist.bbox.numpy()
            gt_label = gt_boxlist.get_field("labels").numpy()
            if gt_label.sum() == 0:
                save_line += str(0)+' '
            else:
                save_line += str(1)+' '
                # gt_bbox의 shape 확인 및 처리
                if gt_bbox.ndim == 1:  # 단일 바운딩 박스인 경우
                    gt_bbox = gt_bbox.reshape(1, -1)  # 2D 배열로 변환
                for gt_idx in range(gt_bbox.shape[0]):
                    x1, y1, x2, y2 = gt_bbox[gt_idx]
                    x1, x2 = np.clip([x1, x2], 0, img_width)
                    y1, y2 = np.clip([y1, y2], 0, img_height)
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)

            # 예측 바운딩 박스 그리기 (이 부분을 수정)
            pred_score = pred_boxlist.get_field("scores").numpy()
            pred_bbox = pred_boxlist.bbox.numpy()
            det_inds = pred_score >= vis_thr
            highscore_score = pred_score[det_inds]
            highscore_bbox = pred_bbox[det_inds]

            if det_fp[image_id][0] > 0:  # FP 케이스
                if highscore_score.size > 0:
                    # 가장 높은 점수를 가진 예측 선택
                    best_pred_idx = np.argmax(highscore_score)
                    x1, y1, x2, y2 = highscore_bbox[best_pred_idx]
                    x1, x2 = np.clip([x1, x2], 0, img_width)
                    y1, y2 = np.clip([y1, y2], 0, img_height)
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 2)  # 파란색으로 FP 표시
                    cv2.putText(img, 'FP: {:.2f}'.format(highscore_score[best_pred_idx]), (int(x1+10), int(y1+10)), cv2.FONT_HERSHEY_PLAIN, 1.2, (255,0,0), 2)
            else:  # TP 또는 다른 케이스
                if highscore_bbox.shape[0] > 0 and gt_bbox.shape[0] > 0:
                    # 기존의 IoU 기반 로직
                    iou_matrix = boxlist_iou(
                        BoxList(torch.from_numpy(highscore_bbox), pred_boxlist.size),
                        BoxList(torch.from_numpy(gt_bbox), gt_boxlist.size)
                    ).numpy()
                    
                    valid_pred_mask = np.max(iou_matrix, axis=1) >= iou_threshold
                    valid_pred_scores = highscore_score[valid_pred_mask]
                    valid_pred_bbox = highscore_bbox[valid_pred_mask]
                    
                    if valid_pred_scores.size > 0:
                        # 가장 높은 점수를 가진 예측 선택
                        best_pred_idx = np.argmax(valid_pred_scores)
                        x1, y1, x2, y2 = valid_pred_bbox[best_pred_idx]
                        x1, x2 = np.clip([x1, x2], 0, img_width)
                        y1, y2 = np.clip([y1, y2], 0, img_height)
                        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,255), 2)
                        cv2.putText(img, '{:.2f}'.format(valid_pred_scores[best_pred_idx]), (int(x1+10), int(y1+10)), cv2.FONT_HERSHEY_PLAIN, 1.2, (0,255,255), 2)

            #결과에 따라 폴더 지정
            if det_tp[image_id][0] > 0:
                save_folder = 'TP'
            elif det_fp[image_id][0] > 0:
                save_folder = 'FP'
            elif det_tn[image_id][0] > 0:
                save_folder = 'TN'
            else:
                save_folder = 'FN'
            
            # 이미지 저장
            save_path = os.path.join(output_folder, save_folder, filename.split('/')[-1] + '.jpg')
            cv2.imwrite(save_path, img)
            
            save_line += str(det_tp[image_id][0])+' '+str(det_fp[image_id][0])+' '+str(det_tn[image_id][0])+' '+str(det_fn[image_id][0])+'\n'
            with open(os.path.join(output_folder, 'result.txt'), 'a+') as save_file:
                save_file.write(save_line)

    return {'Detection': det_evals_dict}