import torchvision
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from torchvision import transforms as T


class ModelUtils:

    # Non-maximum Suppression
    def nms(self, prediction, threshold):
        keep = torchvision.ops.nms(prediction['boxes'], prediction['scores'], threshold)

        final_prediction = prediction
        final_prediction['boxes'] = final_prediction['boxes'][keep]
        final_prediction['scores'] = final_prediction['scores'][keep]
        final_prediction['labels'] = final_prediction['labels'][keep]

        return final_prediction

    def tensorToPIL(self, img):
        return T.transforms.ToPILImage()(img).convert('RGB')

    # Рисование бокса и метки по полученным данным
    def plot_box(self, filename, img, target, classes):
        # Если данные из cuda, то перенесем на cpu
        if target['boxes'].is_cuda:
            target['boxes'] = target['boxes'].cpu()
            target['labels'] = target['labels'].cpu()
            target['scores'] = target['scores'].cpu()
        fig, a = plt.subplots(1)
        plt.axis('off')
        a.imshow(img)
        for box, label_id, score in (zip(target['boxes'], target['labels'], target['scores'])):
            x, y, width, height = box[0], box[1], box[2] - box[0], box[3] - box[1]
            rect = patches.Rectangle((x, y),
                                     width, height,
                                     linewidth=2,
                                     edgecolor='r',
                                     label='Label',
                                     facecolor='none')
            sc = np.around(score.detach().numpy(), 4)
            plt.text(box[0] + 10.0, box[1] + 40.0, classes[label_id] + ': ' + str(sc))
            a.add_patch(rect)
        plt.savefig(filename)
