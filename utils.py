import cv2
import collections
import globals as glo
import torch


def load_model(model_file):
    return torch.load(model_file)


def save_model(model, model_file):
    torch.save(model.state_dict(), model_file)


def preprocess_frame(frame, device):
    frame = torch.from_numpy(frame)
    frame = frame.to(device, dtype=torch.float32)
    frame = frame.unsqueeze(0)
    return frame


def output_video(video_array, size, save_name, evaluate=False, eval_num=0):
    if not evaluate:
        out = cv2.VideoWriter(save_name + "{}.avi".format(glo.EPISODE), cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    else:
        out = cv2.VideoWriter(save_name + "{}.avi".format(eval_num), cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    last_frame = video_array[len(video_array) - 1]

    for x in range(5):
        video_array.append(last_frame)
    for x in range(len(video_array)):
        out.write(video_array[x])
    out.release()

'''
def debug_memory():
    # https://forum.pyro.ai/t/a-clever-trick-to-debug-tensor-memory/556
    print('maxrss = {}'.format(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    tensors = collections.Counter((str(o.device), o.dtype, tuple(o.shape))
                                  for o in gc.get_objects()
                                  if torch.is_tensor(o))
    for line in sorted(tensors.items()):
        print('{}\t{}'.format(*line))
'''