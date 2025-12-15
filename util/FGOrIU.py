import os
import pickle
import torch
from sklearn.decomposition import PCA
import numpy as np

def file_exists(file_path):
    return os.path.exists(file_path)

def move_dict_tensors_to_device(tensor_dict, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in tensor_dict.items()}


def prepare_subspace(
        args,
        model,
        dataloader_forget,
        dataloader_remain,
        device,
        task_id,
        WORK_PATH,
):  

    if args.data_mode == "casia100":
        dim_shape = 512
    elif args.data_mode == "imagenet100":
        dim_shape = 768
    elif args.data_mode == "imagenet1000":
        dim_shape = 768
    elif args.data_mode == "cub200":
        attn_dim = 768
    elif args.data_mode == "omni":
        attn_dim = 768
    else:
        raise NotImplemented

    file_path = './prepared_subspace/{}/setting-start_{}-each_{}/{}/'.format(
        args.data_mode, args.num_of_first_cls, args.per_forget_cls, task_id)


    if file_exists(file_path + 'dr_ffn_cls_pca.pickle'):
        print('>>> prepared_subspace exists')
        if args.feature_subspace_resource == 'ffn' and args.feature_subspace_type == 'cls':
            dr_path = file_path + 'dr_{}_{}_pca.pickle'.format(args.feature_subspace_resource, args.feature_subspace_type)
            df_path = file_path + 'df_{}_{}_pca.pickle'.format(args.feature_subspace_resource, args.feature_subspace_type)
        
        elif args.feature_subspace_resource == 'ffn' and args.feature_subspace_type == 'flatten':
            dr_path = file_path + 'dr_{}_{}_pca.pickle'.format(args.feature_subspace_resource, args.feature_subspace_type)
            df_path = file_path + 'df_{}_{}_pca.pickle'.format(args.feature_subspace_resource, args.feature_subspace_type)
        
        
        elif args.feature_subspace_resource == 'attn' and args.feature_subspace_type == 'cls':
            dr_path = file_path + 'dr_{}_{}_pca.pickle'.format(args.feature_subspace_resource, args.feature_subspace_type)
            df_path = file_path + 'df_{}_{}_pca.pickle'.format(args.feature_subspace_resource, args.feature_subspace_type)
        
        elif args.feature_subspace_resource == 'attn' and args.feature_subspace_type == 'flatten':
            dr_path = file_path + 'dr_{}_{}_pca.pickle'.format(args.feature_subspace_resource, args.feature_subspace_type)
            df_path = file_path + 'df_{}_{}_pca.pickle'.format(args.feature_subspace_resource, args.feature_subspace_type)
        
        else:
            raise NotImplemented

        with open(df_path, 'rb') as f:
            prepare_subspace_df = pickle.load(f)

        with open(dr_path, 'rb') as f:
            prepare_subspace_dr = pickle.load(f)
        
        prepare_subspace_df = move_dict_tensors_to_device(prepare_subspace_df, device)
        prepare_subspace_dr = move_dict_tensors_to_device(prepare_subspace_dr, device)

        return prepare_subspace_df, prepare_subspace_dr
    else:
        os.makedirs(file_path)
        print('>>> build prepared_subspace')

        # dr
        print('dr.......')
        subfeatures_cls_attn_dr, subfeatures_cls_ff_dr, subfeatures_flat_attn_dr, subfeatures_flat_ff_dr = foward2get_subspace(
            dim_shape=dim_shape,
            model=model,
            dataloader=dataloader_remain,
            device=device,
            )

        # df
        print('df.......')
        subfeatures_cls_attn_df, subfeatures_cls_ff_df, subfeatures_flat_attn_df, subfeatures_flat_ff_df = foward2get_subspace(
            dim_shape=dim_shape,
            model=model,
            dataloader=dataloader_forget,
            device=device,
            )

        # sr
        with open(file_path + 'dr_attn_cls_pca.pickle', 'wb') as f:
            pickle.dump(subfeatures_cls_attn_dr, f)

        with open(file_path + 'dr_ffn_cls_pca.pickle', 'wb') as f:
            pickle.dump(subfeatures_cls_ff_dr, f)

        with open(file_path + 'dr_attn_flatten_pca.pickle', 'wb') as f:
            pickle.dump(subfeatures_flat_attn_dr, f)

        with open(file_path + 'dr_ffn_flatten_pca.pickle', 'wb') as f:
            pickle.dump(subfeatures_flat_ff_dr, f)

        # sf
        with open(file_path + 'df_attn_cls_pca.pickle', 'wb') as f:
            pickle.dump(subfeatures_cls_attn_df, f)

        with open(file_path + 'df_ffn_cls_pca.pickle', 'wb') as f:
            pickle.dump(subfeatures_cls_ff_df, f)

        with open(file_path + 'df_attn_flatten_pca.pickle', 'wb') as f:
            pickle.dump(subfeatures_flat_attn_df, f)

        with open(file_path + 'df_ffn_flatten_pca.pickle', 'wb') as f:
            pickle.dump(subfeatures_flat_ff_df, f)

        print('>>> return prepared_subspace')

        if args.feature_subspace_resource == 'ffn' and args.feature_subspace_type == 'cls':
            prepare_subspace_df = subfeatures_cls_ff_df
            prepare_subspace_dr = subfeatures_cls_ff_dr
        
        elif args.feature_subspace_resource == 'ffn' and args.feature_subspace_type == 'flatten':
            prepare_subspace_df = subfeatures_flat_ff_df
            prepare_subspace_dr = subfeatures_flat_ff_dr

        elif args.feature_subspace_resource == 'attn' and args.feature_subspace_type == 'cls':
            prepare_subspace_df = subfeatures_cls_attn_df
            prepare_subspace_dr = subfeatures_cls_attn_dr

        elif args.feature_subspace_resource == 'attn' and args.feature_subspace_type == 'flatten':
            prepare_subspace_df = subfeatures_flat_attn_df
            prepare_subspace_dr = subfeatures_flat_attn_dr

        else:
            raise NotImplemented
    
        prepare_subspace_df = move_dict_tensors_to_device(prepare_subspace_df, device)
        prepare_subspace_dr = move_dict_tensors_to_device(prepare_subspace_dr, device)

    return prepare_subspace_df, prepare_subspace_dr

def foward2get_subspace(
        dim_shape,
        model,
        dataloader,
        device,
        ):
    model.eval()

    subfeatures_cls_attn = {}  
    subfeatures_flat_attn = {} 
    subfeatures_cls_ff = {}  
    subfeatures_flat_ff = {} 

    with torch.no_grad():
        for inputs_remain, labels_remain in iter(dataloader):
            inputs_remain = inputs_remain.to(device)
            labels_remain = labels_remain.to(device)
            logit, subfeatures, embeds = model(inputs_remain.float(), labels_remain, prepare_feature=True)

            num_layers = len(subfeatures) // 2  

            for i in range(num_layers):

                attn_output = subfeatures[f"layer_{i}_attn"] 
                ff_output = subfeatures[f"layer_{i}_ff"]  


                cls_token_attn = attn_output[:, 0, :] 
                cls_token_ff = ff_output[:, 0, :] 

                flat_attn = attn_output.view(attn_output.shape[0], -1) 
                flat_ff = ff_output.view(ff_output.shape[0], -1)  

                if f"layer_{i}" not in subfeatures_cls_attn:
                    subfeatures_cls_attn[f"layer_{i}"] = []
                    subfeatures_flat_attn[f"layer_{i}"] = []
                    subfeatures_cls_ff[f"layer_{i}"] = []
                    subfeatures_flat_ff[f"layer_{i}"] = []

                subfeatures_cls_attn[f"layer_{i}"].append(cls_token_attn)
                subfeatures_flat_attn[f"layer_{i}"].append(flat_attn)
                subfeatures_cls_ff[f"layer_{i}"].append(cls_token_ff)
                subfeatures_flat_ff[f"layer_{i}"].append(flat_ff)

        num_layers = len(subfeatures_cls_attn)

        for i in range(num_layers):
            subfeatures_cls_ff[f"layer_{i}"] = torch.cat(subfeatures_cls_ff[f"layer_{i}"], dim=0).cpu() 


            subfeatures_cls_attn[f"layer_{i}"] = subfeatures_cls_ff[f"layer_{i}"]
            subfeatures_flat_attn[f"layer_{i}"] = subfeatures_cls_ff[f"layer_{i}"]
            subfeatures_flat_ff[f"layer_{i}"] = subfeatures_cls_ff[f"layer_{i}"]
    

        num_layers = len(subfeatures_cls_attn)

        print("============================================================")
        print('subfeatures_cls_attn')
        subfeatures_cls_attn = update_memory_prefix(subfeatures_cls_attn, features={}) 
        print("============================================================")
        print('subfeatures_cls_ff')
        subfeatures_cls_ff = update_memory_prefix(subfeatures_cls_ff, features={})

        print("============================================================")
        print('subfeatures_flat_attn')
        subfeatures_flat_attn = update_memory_prefix(subfeatures_flat_attn, features={})
        print("============================================================")
        print('subfeatures_flat_ff')
        subfeatures_flat_ff = update_memory_prefix(subfeatures_flat_ff, features={})
        
        return subfeatures_cls_attn, subfeatures_cls_ff, subfeatures_flat_attn, subfeatures_flat_ff

 
import torch

def update_memory_prefix(represent, threshold=0.99, features=None):
    if features is None:
        features = {}

    for layer in represent:
        representation = represent[layer]  

        gram_matrix = representation.T @ representation

        if layer not in features or features[layer] is None:

            U, S, Vh = torch.linalg.svd(gram_matrix, full_matrices=False)
            total_energy = (S ** 2).sum()
            cumulative_energy = torch.cumsum(S ** 2, dim=0) / total_energy

            r = torch.searchsorted(cumulative_energy, threshold).item() + 1
            
            if r == 0:
                raise NameError

            bases = Vh[:r, :].T.to(representation.device)

            features[layer] = bases

        else:
            P = features[layer] 
            
            projected_representation = representation - P @ P.T @ representation
            
            U, S, Vh = torch.linalg.svd(projected_representation, full_matrices=False)
            
            total_energy = (S ** 2).sum()
            cumulative_energy = torch.cumsum(S ** 2, dim=0) / total_energy
            accumulated_energy = 1 - (total_energy / (S ** 2).sum()) 
            
            r = 0
            for i in range(len(cumulative_energy)):
                if accumulated_energy < threshold:
                    accumulated_energy += cumulative_energy[i]
                    r += 1
                else:
                    break

            if r > 0:
                Vh_new = Vh[:r, :]
                P = torch.cat([P, Vh_new.T], dim=1) 

            if P.shape[1] > P.shape[0]:
                P = P[:, :P.shape[0]]

            features[layer] = P.to(representation.device)

    return features




def filter_feature(
        features: dict,
        f_resource,
        f_type,
        ):

    num_layers = len(features) // 2
    return_features = {}

    if f_resource == 'ffn' and f_type == 'cls':
        for i in range(num_layers):
            ff_output = features[f"layer_{i}_ff"] 
            cls_token_ff = ff_output[:, 0, :]  
            return_features[f"layer_{i}"] = F.normalize(cls_token_ff, p=2, dim=1)
        
        return return_features
    
    elif f_resource == 'ffn' and f_type == 'flatten':
        for i in range(num_layers):
            ff_output =  F.normalize(features[f"layer_{i}_ff"], p=2, dim=2)
            flat_attn = ff_output.view(-1, ff_output.shape[2]) 
            return_features[f"layer_{i}"] = flat_attn
        
        return return_features
    
    elif f_resource == 'attn' and f_type == 'cls':
        for i in range(num_layers):
            attn_output = features[f"layer_{i}_attn"]
            cls_token_attn = attn_output[:, 0, :]  
            return_features[f"layer_{i}"] = F.normalize(cls_token_attn, p=2, dim=1)
        
        return return_features

    elif f_resource == 'attn' and f_type == 'flatten':
        for i in range(num_layers):
            attn_output = F.normalize(features[f"layer_{i}_attn"], p=2, dim=2)
            flat_attn = attn_output.view(-1, ff_output.shape[2])  
            return_features[f"layer_{i}"] = flat_attn
        
        return return_features
    
    else:
        raise NotImplemented


import torch
import torch.nn as nn
import torch.nn.functional as F

def gumbel_softmax(logits, tau=1.0, hard=True):
    """Gumbel-Softmax 采样"""
    gumbels = -torch.empty_like(logits).exponential_().log() 
    y = logits + gumbels
    y_soft = F.softmax(y / tau, dim=-1) 
    
    if hard:

        y_hard = F.one_hot(y_soft.argmax(dim=-1), num_classes=y.shape[-1]).float()
        y = (y_hard - y_soft).detach() + y_soft 
    else:
        y = y_soft

    return y

class SparseSelector(nn.Module):

    def __init__(self, num_layers, tau=1.0):
        super().__init__()
        self.logits = nn.Parameter(torch.ones(num_layers, 2))  
        self.tau = tau  

    def forward(self):
        probs = gumbel_softmax(self.logits, tau=self.tau, hard=True) 
        return probs[:, 1]  


def nuclear_norm_loss(features, feature_spaces, selector, complement=False):

    loss = 0.0
    for i, (layer, F) in enumerate(features.items()):
        P = feature_spaces[layer]  

        P = P.to(torch.float32)
        F = F.to(torch.float32)
        if complement:
            
            projection = P @ (P.T @ F.T)  
            residual = F - projection.T 
            l2_norm = torch.norm(residual, p=2) 
            if selector is not None:
                loss += selector[i] * l2_norm
            else:
                loss += l2_norm 

        else:

            F_T_P = F @ P 
            nuclear_norm = torch.linalg.norm(F_T_P, ord='nuc') 
            if selector is not None:
                loss += selector[i] * nuclear_norm 
            else:
                loss += nuclear_norm 
            
    return loss



def nuclear_norm_loss(features, feature_spaces, selector, complement=False, imagenet100=False):
    loss = 0.0
    layers = list(features.keys())  

    if imagenet100:

        last_layer = layers[-1]
        F = features[last_layer]
        P = feature_spaces[last_layer]
        
        P = P.to(torch.float32)
        F = F.to(torch.float32)
        
        if complement:
            projection = P @ (P.T @ F.T) 
            residual = F - projection.T 
            l2_norm = torch.norm(residual, p=2)
            if selector is not None:
                loss += selector[-1] * l2_norm 
            else:
                loss += l2_norm

        else:

            F_T_P = F @ P 
            nuclear_norm = torch.linalg.norm(F_T_P, ord='nuc') 
            if selector is not None:
                loss += selector[-1] * nuclear_norm 
            else:
                loss += nuclear_norm

    else:

        for i, (layer, F) in enumerate(features.items()):
            P = feature_spaces[layer] 

            P = P.to(torch.float32)
            F = F.to(torch.float32)
            if complement:

                projection = P @ (P.T @ F.T)  
                residual = F - projection.T  
                l2_norm = torch.norm(residual, p=2)  
                if selector is not None:
                    loss += selector[i] * l2_norm 
                else:
                    loss += l2_norm 

            else:

                F_T_P = F @ P  
                nuclear_norm = torch.linalg.norm(F_T_P, ord='nuc')  
                if selector is not None:
                    loss += selector[i] * nuclear_norm  
                else:
                    loss += nuclear_norm  

    return loss


def compute_complement_space(P):

    U, S, V = torch.linalg.svd(P)
    P_complement = V[:, len(S):]  
    return P_complement








def project_onto_space(grad, space):

    grad = grad.to(torch.float32)
    space = space.to(torch.float32)

    projection = grad @ space 
    return projection @ space.T  

def project_onto_complement_space(grad, space):

    projected_grad = project_onto_space(grad, space)

    complement_grad = grad - projected_grad
    return complement_grad 

def grad_proj(args, model, device, coef_df, coef_dr, sf, sr, layer_idx):


    if args.data_mode == "casia100":
        attn_dim = 512
    elif args.data_mode == "imagenet100":
        attn_dim = 768 
    elif args.data_mode == "imagenet1000":
        attn_dim = 768
    elif args.data_mode == "cub200":
        attn_dim = 768
    elif args.data_mode == "omni":
        attn_dim = 768
    else:
        raise NotImplemented

    lora_rank = args.lora_rank
    for name, param in model.named_parameters():
        if 'lora' not in name:  
            continue

        grad = param.grad.to(device)  


        layer_name = name.split('.')[layer_idx]  
        layer_sf = sf[f"layer_{layer_name}"].to(device) 
        layer_sr = sr[f"layer_{layer_name}"].to(device) 


        if grad.shape[0] == lora_rank and grad.shape[1] == attn_dim:  

            if args.backward_disturbing_on_DF and args.backward_denoising_on_DR:
                projected_grad = project_onto_space(grad, layer_sf)
                complement_projected_grad = project_onto_complement_space(grad, layer_sr)

                param.grad = coef_df * projected_grad + coef_dr * complement_projected_grad
            elif args.backward_disturbing_on_DF and not args.backward_denoising_on_DR:

                projected_grad = project_onto_space(grad, layer_sf)

                param.grad = coef_df * projected_grad
            elif not args.backward_disturbing_on_DF and args.backward_denoising_on_DR:

                complement_projected_grad = project_onto_complement_space(grad, layer_sr)

                param.grad = coef_dr * complement_projected_grad
            elif not args.backward_disturbing_on_DF and not args.backward_denoising_on_DR:

                pass
            else:
                raise NotImplemented
        elif grad.shape[0] == attn_dim and grad.shape[1] == lora_rank: 

            old_shape_sub = grad.shape
            grad = grad.reshape(old_shape_sub[1], old_shape_sub[0])
            if args.backward_disturbing_on_DF and args.backward_denoising_on_DR:
                projected_grad = project_onto_space(grad, layer_sf)
                complement_projected_grad = project_onto_complement_space(grad, layer_sr)

                param.grad = coef_df * projected_grad.reshape(old_shape_sub) + coef_dr * complement_projected_grad.reshape(old_shape_sub)
            elif args.backward_disturbing_on_DF and not args.backward_denoising_on_DR:

                projected_grad = project_onto_space(grad, layer_sf)

                param.grad = coef_df * projected_grad.reshape(old_shape_sub)
            elif not args.backward_disturbing_on_DF and args.backward_denoising_on_DR:

                complement_projected_grad = project_onto_complement_space(grad, layer_sr)

                param.grad = coef_dr * complement_projected_grad.reshape(old_shape_sub)
            elif not args.backward_disturbing_on_DF and not args.backward_denoising_on_DR:

                pass
            else:
                raise NotImplemented


        elif grad.shape[0] == attn_dim*4 and grad.shape[1] == lora_rank: 

            old_shape = grad.shape
            grad_resized = grad.view(attn_dim, -1)


            if args.backward_disturbing_on_DF and args.backward_denoising_on_DR:
                projected_grad = project_onto_space(grad_resized.T, layer_sf)
                complement_projected_grad = project_onto_complement_space(grad_resized.T, layer_sr)

                post_grad = coef_df * projected_grad + coef_dr * complement_projected_grad
            elif args.backward_disturbing_on_DF and not args.backward_denoising_on_DR:

                projected_grad = project_onto_space(grad_resized.T, layer_sf)

                post_grad = coef_df * projected_grad
            elif not args.backward_disturbing_on_DF and args.backward_denoising_on_DR:

                complement_projected_grad = project_onto_complement_space(grad_resized.T, layer_sr)
                # print('complement_projected_grad', complement_projected_grad.shape)

                post_grad = coef_dr * complement_projected_grad
            elif not args.backward_disturbing_on_DF and not args.backward_denoising_on_DR:

                post_grad = grad_resized
                pass
            else:
                raise NotImplemented
        
        elif grad.shape[0] == lora_rank and grad.shape[1] == attn_dim*4:  

            old_shape = grad.shape
            grad_resized = grad.view(attn_dim, -1)


            if args.backward_disturbing_on_DF and args.backward_denoising_on_DR:
                projected_grad = project_onto_space(grad_resized.T, layer_sf)
                complement_projected_grad = project_onto_complement_space(grad_resized.T, layer_sr)

                post_grad = coef_df * projected_grad + coef_dr * complement_projected_grad
            elif args.backward_disturbing_on_DF and not args.backward_denoising_on_DR:

                projected_grad = project_onto_space(grad_resized.T, layer_sf)

                post_grad = coef_df * projected_grad
            elif not args.backward_disturbing_on_DF and args.backward_denoising_on_DR:

                complement_projected_grad = project_onto_complement_space(grad_resized.T, layer_sr)
                # print('complement_projected_grad', complement_projected_grad.shape)

                post_grad = coef_dr * complement_projected_grad
            elif not args.backward_disturbing_on_DF and not args.backward_denoising_on_DR:

                post_grad = grad_resized
                pass
            else:
                raise NotImplemented
            


            param.grad = post_grad.reshape(old_shape)  
        else:

            continue

    return model
