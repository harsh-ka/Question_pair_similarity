from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
def plot_confusion_matrix_new(y_true,y_pred):
  C=confusion_matrix(y_true,y_pred,)
  plt.figure(figsize=(15,4))
  #C is a Confusion matrix
  #C=[[3,4],     [[TN,FN],
  #                [FP,TP ]]
  #   [2,7]]
 
  labels=[0,1]
  ax1=plt.subplot(1,3,1)
  sns.heatmap(C,annot=True,fmt='d',xticklabels=labels,yticklabels=labels,ax=ax1,cbar=False)
  ax1.set_xlabel('Actual class',)
  ax1.set_ylabel('Predicted_class')
  ax1.set_title('Confusion_matrix')
  


  #Calculation of the precision score
  #   [[TN/TN+FN,FN/TN+FN],
  #    [FP/FP+TP,TP/FP+TP]]
  #In two dimensional array axis=0 corresponds to col where as axis 1 corresponds to row
  #C=[[1,4],   np.sum(C,axis=1)=[5,7]  C.T=[[1,2],] C.T/np.sum(X.axis=1)=[[1/5,4/7],]
  #    [2,3]]                               [4,3]]                         [2/5,3/7]]
  # Then we can take the transpose of the final values
  precision_matrix=C/np.sum(C,axis=0)
  labels=[0,1]
  ax2=plt.subplot(1,3,2)
  sns.heatmap(precision_matrix,fmt='f',annot=True,xticklabels=labels,yticklabels=labels,ax=ax2,cbar=False)
  ax2.set_xlabel('Actual class',)
  ax2.set_ylabel('Predicted_class')
  ax2.set_title('Precision matrix')
 

  #recal_matrix=
  #C=[[TN,FN],]  [[TN/TN+FP,FN/FN+TP],]
  #   [FP,TP]]    [FP/TN+FP ....]
  #
  recall_matrix=(C.T/np.sum(C,axis=1)).T
  lables=[0,1]
  ax3=plt.subplot(1,3,3)
  sns.heatmap(recall_matrix,fmt='f',annot=True,xticklabels=labels,yticklabels=labels,ax=ax3,cbar=False)
  plt.xlabel('Actual class',)
  plt.ylabel('Predicted_class')
  plt.title('Recall matrix')
  plt.show()
  