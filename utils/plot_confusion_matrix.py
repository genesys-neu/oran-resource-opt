import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np


t124 = np.array([[36731,  1442, 1372], [3476, 55642,  9874], [5818,  1191, 88209]])
t132 = np.array([[40319, 1205, 9159], [4510, 55299, 7502], [6384, 877, 85819]])
t140 = np.array([[39328,  1916,  8153], [3752, 55730,  6078], [4262,  1316, 84544]])
t148 = np.array([[40756,  1786,  5620], [4506, 55130,  4231], [3907,  1304, 82462]])
data = t148
name = 't148'
print(data)
data_percent = np.around(data / data.sum(axis=1)[:, np.newaxis], decimals=4)
print(data_percent)


# Plot the confusion matrix
plt.figure(figsize=(6, 3))
sns.heatmap(data_percent, annot=True, fmt='.2%', cmap='Blues', cbar=False, annot_kws={"size": 24})
plt.xlabel('Predicted Label', fontsize=18)
plt.ylabel('True Label', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

# Remove internal gridlines
plt.grid(False)

plt.gca().spines['top'].set_visible(True)  # show top line
plt.gca().spines['right'].set_visible(True)  # show right line
plt.gca().spines['left'].set_visible(True)  # show top line
plt.gca().spines['bottom'].set_visible(True)  # show right line
plt.tight_layout()
# plt.title(f'Confusion Matrix - Model {model_name}')
# Save the confusion matrix plot as a .png file
# plt.savefig(f"dt.png")
plt.savefig(f"{name}.pdf", format='pdf')
plt.close()
