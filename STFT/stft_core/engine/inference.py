# Copyright (c) SenseTime Research and its affiliates. All Rights Reserved.
import logging
import time
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from stft_core.structures.image_list import to_image_list

from stft_core.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str
from .bbox_aug import im_detect_bbox_aug
from stft_core.structures.bounding_box import BoxList


def compute_on_dataset(model, dataset, data_loader, device, bbox_aug, method, timer=None):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for i, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch

        with torch.no_grad():
            if timer:
                timer.tic()
            if bbox_aug:
                output = im_detect_bbox_aug(model, images, device)
            else:
                if method in ("base", "cvc_image"):
                    images = images.to(device)
                elif method in ("rdn", "mega", "fgfa", "stft", "cvc_fgfa", "cvc_mega", "cvc_rdn", "cvc_stft"):
                    images["cur"] = images["cur"].to(device)
                    for key in ("ref", "ref_l", "ref_m", "ref_g"):
                        if key in images.keys():
                            images[key] = [img.to(device) for img in images[key]]
                else:
                    raise ValueError("method {} not supported yet.".format(method))

                output = model(images)
                    
            if timer:
                if not device.type == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()

            output = [o.to(cpu_device) for o in output]

        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("stft_core.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        cfg,
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        motion_specific=False,
        box_only=False,
        bbox_aug=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
        visulize=True,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("stft_core.inference")
    dataset = data_loader.dataset
    if is_main_process():
        logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(model, dataset, data_loader, device, bbox_aug, cfg.MODEL.VID.METHOD, inference_timer)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    if is_main_process():
        logger.info(
            "Total run time: {} ({} s / img per device, on {} devices)".format(
                total_time_str, total_time * num_devices / len(dataset), num_devices
            )
        )
    total_infer_time = get_time_str(inference_timer.total_time)
    if is_main_process():
        logger.info(
            "Model inference time: {} ({} s / img per device, on {} devices)".format(
                total_infer_time,
                inference_timer.total_time * num_devices / len(dataset),
                num_devices,
            )
        )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        motion_specific=motion_specific,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
        visulize=visulize,
        vis_thr=cfg.TEST.VIS_THR,
        iou_threshold=cfg.TEST.VIS_IOU,
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)


def inference_no_model(
        data_loader,
        iou_types=("bbox",),
        motion_specific=False,
        box_only=False,
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
):
    dataset = data_loader.dataset

    predictions = torch.load(os.path.join(output_folder, "predictions.pth"))
    print("prediction loaded.")

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        motion_specific=motion_specific,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
        vis_thr=0.5,
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)


def process_image(cfg, model, image_path, output_folder, device="cuda"):
    # 이미지 로드 및 전처리
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 이미지 크기 조정 (필요한 경우)
    if cfg.INPUT.MIN_SIZE_TEST != -1:
        print('!!!!!!!!!!!!!!!!resized!!!!!!!!!!!!!!!')
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        h, w = image.shape[:2]
        if max(h, w) > max_size:
            scale = float(max_size) / float(max(h, w))
            image = cv2.resize(image, None, fx=scale, fy=scale)
        elif min(h, w) < min_size:
            scale = float(min_size) / float(min(h, w))
            image = cv2.resize(image, None, fx=scale, fy=scale)

    # 이미지를 텐서로 변환
    image = torch.from_numpy(image.transpose(2, 0, 1)).float()
    image = image.to(device)
    image = to_image_list(image, cfg.DATALOADER.SIZE_DIVISIBILITY)

    # 모델 추론
    with torch.no_grad():
        output = model(image)

    # 결과 후처리
    output = output[0].to(torch.device("cpu"))
    boxes = output.bbox.tolist()
    labels = output.get_field("labels").tolist()
    scores = output.get_field("scores").tolist()

    # 결과 저장
    result = {
        "boxes": boxes,
        "labels": labels,
        "scores": scores
    }

    # 결과를 파일로 저장
    image_name = os.path.basename(image_path)
    output_path = os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}_result.json")
    with open(output_path, "w") as f:
        json.dump(result, f)

    # 시각화 (선택적)
    if cfg.TEST.VISUALIZE:
        visualize_result(image_path, result, output_folder)

    return result

def visualize_result(image_path, result, output_folder):
    image = cv2.imread(image_path)
    for box, label, score in zip(result["boxes"], result["labels"], result["scores"]):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{label}: {score:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    output_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(image_path))[0]}_vis.jpg")
    cv2.imwrite(output_path, image)
