import torch
import torch.nn.functional as F
from utils import utils
import numpy as np

# class_num = 4
# concept_embeddings = torch.nn.Embedding(class_num, 64)
# concept = torch.tensor([0,1,2,3])
# x = torch.zeros((64, 64))
# concepts = concept_embeddings(concept)
# zero = concepts[0]
# if len(x.size()) < len(zero.size()):
#     x = x.expand_as(zero)
# else:
#     zero = zero.expand_as(x)
#
# vector = torch.cat((x, zero), dim=-1)
# print(vector)



x = torch.randint(999,(4,))
print(x)


# def similarity(vector1, vector2, sigmoid=True):
#     result = F.cosine_similarity(vector1, vector2, dim=-1)
#     result = result * sim_scale
#     if sigmoid:
#         return result.sigmoid()
#     return result
# sim_scale = 10
# vector1 = torch.nn.Parameter(utils.numpy_to_torch(
#             np.random.uniform(0, 1, size=[1, 64]).astype(np.float32)), requires_grad=False)
# vector2 = torch.nn.Parameter(utils.numpy_to_torch(
#             np.random.uniform(0, 1, size=[1, 64]).astype(np.float32)), requires_grad=False)
#
# pre = similarity(vector1, vector2)
# print(pre)

