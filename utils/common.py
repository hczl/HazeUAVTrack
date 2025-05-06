import sys

import importlib


import cv2
import numpy as np

def call_function(method_name, module_prefix, *args):
    """Dynamically imports and calls a processing function (core version)."""
    module_name = f'{module_prefix}.{method_name}'
    spec = importlib.util.find_spec(module_name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    func = getattr(module, method_name)
    return func(*args)


def preview_batch_with_boxes(loader, class_names=None, window_name='Preview'):
    import matplotlib.colors as mcolors
    color_list = list(mcolors.TABLEAU_COLORS.values())

    for i, (images, targets, _) in enumerate(loader):
        images = images[:2]  # 仅前两张图像
        targets = targets[:2]
        for idx in range(len(images)):
            img = images[idx].permute(1, 2, 0).cpu().numpy() * 255
            img = img.astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            for box in targets[idx]:
                # 解包标签格式
                # [frame_index, target_id, x1, y1, w, h, out-of-view, occlusion, category]
                _, _, x1, y1, w, h, _, _, category = box.tolist()
                x1, y1, w, h = map(int, [x1, y1, w, h])
                x2 = x1 + w
                y2 = y1 + h
                cls_id = int(category)
                label = f"{class_names[cls_id]}" if class_names else f"Class {cls_id}"
                color = color_list[cls_id % len(color_list)]
                bgr_color = tuple(int(255 * c) for c in mcolors.to_rgb(color))

                cv2.rectangle(img, (x1, y1), (x2, y2), bgr_color, 2)
                cv2.putText(img, label, (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, bgr_color, 1, cv2.LINE_AA)

            cv2.imshow(f"{window_name} - Image {idx+1}", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break  # 只展示一个 batch
