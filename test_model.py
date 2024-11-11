import torch.optim
from Load_Dataset import ValGenerator, ImageToImage2D
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")
import Config as config
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from nets.UC_Res_Swin_ECA import UC_Res_Swin_ECA
from nets.skip3 import skip3
from nets.skip5 import skip5
from utils import *
import cv2


import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix, accuracy_score

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, classes, save_path, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        title = 'Normalized confusion matrix' if normalize else 'Confusion matrix, without normalization'

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(classes)),
           yticks=np.arange(len(classes)),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], '.2f' if normalize else 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()




def show_image_with_dice(predict_save, labs, save_path):

    tmp_lbl = (labs).astype(np.float32)
    tmp_3dunet = (predict_save).astype(np.float32)
    dice_pred = 2 * np.sum(tmp_lbl * tmp_3dunet) / (np.sum(tmp_lbl) + np.sum(tmp_3dunet) + 1e-5)
    # dice_show = "%.3f" % (dice_pred)
    iou_pred = jaccard_score(tmp_lbl.reshape(-1),tmp_3dunet.reshape(-1))
    # fig, ax = plt.subplots()
    # plt.gca().add_patch(patches.Rectangle(xy=(4, 4),width=120,height=20,color="white",linewidth=1))
    if config.task_name == "MoNuSeg":
        predict_save = cv2.pyrUp(predict_save,(448,448))
        predict_save = cv2.resize(predict_save,(224,224))
        # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #定义一个核
        # predict_save = cv2.filter2D(predict_save, -1, kernel=kernel)
        cv2.imwrite(save_path,predict_save * 255)
    else:
        cv2.imwrite(save_path,predict_save * 255)
    # plt.imshow(predict_save * 255,cmap='gray')
    # plt.text(x=10, y=24, s="Dice:" + str(dice_show), fontsize=5)
    # plt.axis("off")
    # remove the white borders
    # height, width = predict_save.shape
    # fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    # plt.margins(0, 0)
    # plt.savefig(save_path, dpi=2000)
    # plt.close()
    return dice_pred, iou_pred

def vis_and_save_heatmap(model, input_img, img_RGB, labs, vis_save_path, dice_pred, dice_ens):
    model.eval()

    output = model(input_img.cuda())
    pred_class = torch.where(output>0.5,torch.ones_like(output),torch.zeros_like(output))
    predict_save = pred_class[0].cpu().data.numpy()
    predict_save = np.reshape(predict_save, (config.img_size, config.img_size))
    dice_pred_tmp, iou_tmp = show_image_with_dice(predict_save, labs, save_path=vis_save_path+'_pre_'+model_type+'.jpg')
    return dice_pred_tmp, iou_tmp

# def calculate_metrics(true_labels, predictions):
#     accuracy = accuracy_score(true_labels, predictions)
#     recall = recall_score(true_labels, predictions, average='macro')
#     precision = precision_score(true_labels, predictions, average='macro')
#     return accuracy, recall, precision

def calculate_metrics(y_true, y_pred):
    # 计算准确率
    accuracy = accuracy_score(y_true, y_pred)

    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # 计算敏感性和特异性
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return accuracy, sensitivity, specificity

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    test_session = config.test_session
    if config.task_name == "GlaS":
        test_num = 80
        model_type = config.model_name
        model_path = "./GlaS/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth"

    elif config.task_name == "MoNuSeg":
        test_num = 14
        model_type = config.model_name
        model_path = "./MoNuSeg/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth"
        
        
    elif config.task_name == "GlaS":
        test_num = 80
        model_type = config.model_name
        model_path = "./GlaS/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth"
        
        
    elif config.task_name == "DRIVE":
        test_num = 20
        model_type = config.model_name
        model_path = "./DRIVE/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth"
        
    elif config.task_name == "CHASEDB":
        test_num = 8
        model_type = config.model_name
        model_path = "./CHASEDB/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth"
        
     elif config.task_name == "stare":
        test_num = 4
        model_type = config.model_name
        model_path = "./stare/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth"



    save_path  = config.task_name +'/'+ model_type +'/' + test_session + '/'
    vis_path = "./" + config.task_name + '_preImg/'
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)

    checkpoint = torch.load(model_path, map_location='cuda')


    if model_type == 'UC_Res_Swin_ECA':
        config_vit = config.get_CTranS_config()
        model = UC_Res_Swin_ECA(config_vit,n_channels=config.n_channels,n_classes=config.n_labels)
        
    elif model_type == 'skip3':
        config_vit = config.get_CTranS_config()
        model = skip3(config_vit,n_channels=config.n_channels,n_classes=config.n_labels)
        
    elif model_type == 'skip5':
        config_vit = config.get_CTranS_config()
        model = skip5(config_vit,n_channels=config.n_channels,n_classes=config.n_labels)



    else: raise TypeError('Please enter a valid name for the model type')

    model = model.cuda()
    if torch.cuda.device_count() > 1:
        print ("Let's use {0} GPUs!".format(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=[0,1,2,3])
    model.load_state_dict(checkpoint['state_dict'])
    print('Model loaded !')
    tf_test = ValGenerator(output_size=[config.img_size, config.img_size])
    test_dataset = ImageToImage2D(config.test_dataset, tf_test,image_size=config.img_size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    dice_pred = 0.0
    iou_pred = 0.0
    dice_ens = 0.0

    with tqdm(total=test_num, desc='Test visualize', unit='img', ncols=70, leave=True) as pbar:
        for i, (sampled_batch, names) in enumerate(test_loader, 1):
            test_data, test_label = sampled_batch['image'], sampled_batch['label']
            arr=test_data.numpy()
            arr = arr.astype(np.float32())
            lab=test_label.data.numpy()
            img_lab = np.reshape(lab, (lab.shape[1], lab.shape[2])) * 255
            fig, ax = plt.subplots()
            plt.imshow(img_lab, cmap='gray')
            plt.axis("off")
            height, width = config.img_size, config.img_size
            fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)
            name = names[0]  # 获取元组中的第一个元素
            plt.savefig(vis_path+str(name[:-4])+"_lab.jpg", dpi=300)
            plt.close()
            input_img = torch.from_numpy(arr)
            dice_pred_t,iou_pred_t = vis_and_save_heatmap(model, input_img, None, lab,
                                                          vis_path+str(name[:-4]),
                                               dice_pred=dice_pred, dice_ens=dice_ens)
            dice_pred+=dice_pred_t
            iou_pred+=iou_pred_t
            torch.cuda.empty_cache()
            pbar.update()
    print ("dice_pred",dice_pred/test_num)
    print ("iou_pred",iou_pred/test_num)
    
    
    model.eval()

    total_accuracy = 0
    total_sensitivity = 0
    total_specificity = 0
    count = 0

    with torch.no_grad(), tqdm(total=test_num, desc='Evaluating', unit='img', ncols=70, leave=True) as pbar:
        for i, (sampled_batch, names) in enumerate(test_loader, 1):
            test_data, test_label = sampled_batch['image'], sampled_batch['label']
            input_img = test_data.cuda()
            true_labels = test_label.data.numpy().flatten()  # 真实标签

            output = model(input_img)
            pred_class = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output))
            predictions = pred_class.cpu().data.numpy().flatten()  # 预测结果

            # 计算指标
            accuracy, sensitivity, specificity = calculate_metrics(true_labels, predictions)
            total_accuracy += accuracy
            total_sensitivity += sensitivity
            total_specificity += specificity
            count += 1

            pbar.update()

      # 计算平均指标
    avg_accuracy = total_accuracy / count
    avg_sensitivity = total_sensitivity / count
    avg_specificity = total_specificity / count

    print(f"Average Accuracy: {avg_accuracy:.5f}")
    print(f"Average Sensitivity: {avg_sensitivity:.5f}")
    print(f"Average Specificity: {avg_specificity:.5f}")
    
    
    model.eval()
    with torch.no_grad(), tqdm(total=test_num, desc='Evaluating', unit='img', ncols=70, leave=True) as pbar:
        for i, (sampled_batch, names) in enumerate(test_loader, 1):
            test_data, test_label = sampled_batch['image'], sampled_batch['label']
            input_img = test_data.cuda()
            true_labels = test_label.squeeze().cpu().numpy().flatten()  # 真实标签
            output = model(input_img)
            predictions = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output)).squeeze().cpu().numpy().flatten()  # 预测结果

            # 可视化并保存混淆矩阵
            class_labels = ['Positive', 'Negative']  # 示例类别标签
            save_path = '/root/autodl-tmp/UCTransNet-main/confusion_matrix.png'
            plot_confusion_matrix(true_labels, predictions, class_labels, save_path, normalize=True)

            # 计算指标
            accuracy, sensitivity, specificity = calculate_metrics(true_labels, predictions)
            total_accuracy += accuracy
            total_sensitivity += sensitivity
            total_specificity += specificity
            count += 1

            pbar.update()





