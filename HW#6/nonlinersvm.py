from sklearn.model_selection import train_test_split

# Split the moons dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_moons, y_moons, test_size=0.2, random_state=42)

# Print the shapes of the split datasets
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)
from sklearn.preprocessing import StandardScaler

# Create a StandardScaler object
scaler = StandardScaler()

# Fit the scaler to the training set and transform it
X_train_scaled = scaler.fit_transform(X_train)

# Transform the test set using the scaler fitted to the training set
X_test_scaled = scaler.transform(X_test)
from sklearn.svm import SVC
from scipy.stats import randint, uniform

# Create a polynomial kernel SVM model
poly_model = SVC(kernel='poly')

# Define the hyperparameters to search over
poly_params = {
    'C': uniform(0, 20),  # C parameter
    'degree': randint(low=2, high=5),  # Degree of the polynomial kernel
    'coef0': randint(low=0, high=5)  # Independent term in kernel function
}

# Tune the polynomial kernel SVM model
best_poly_model = tune(poly_model, poly_params, X_train_scaled, y_train)
# Create an RBF kernel SVM model
rbf_model = SVC(kernel='rbf')

# Define the hyperparameters to search over
rbf_params = {
    'C': uniform(0, 20),  # C parameter
    'gamma': uniform(0, 1)  # Kernel coefficient for RBF
}

# Tune the RBF kernel SVM model
best_rbf_model = tune(rbf_model, rbf_params, X_train_scaled, y_train)
from sklearn.metrics import classification_report, confusion_matrix

# Predict class labels for the test set using the best polynomial model
y_pred_poly = best_poly_model.predict(X_test_scaled)

# Print classification report for the polynomial model
print("Classification Report for Polynomial Kernel:")
print(classification_report(y_test, y_pred_poly))

# Calculate and print confusion matrix for the polynomial model
print("Confusion Matrix for Polynomial Kernel:")
print(confusion_matrix(y_test, y_pred_poly))

# Predict class labels for the test set using the best RBF model
y_pred_rbf = best_rbf_model.predict(X_test_scaled)

# Print classification report for the RBF model
print("Classification Report for RBF Kernel:")
print(classification_report(y_test, y_pred_rbf))

# Calculate and print confusion matrix for the RBF model
print("Confusion Matrix for RBF Kernel:")
print(confusion_matrix(y_test, y_pred_rbf))

def plot_svm(clf, X_train, y_train, X_test, y_test, C_value):
    """
    Generate a simple plot of SVM including the decision boundary, margin, and its training/test data

    Parameters
    ----------
    clf: your classifier handle (after training)
    X: feature matrix shape(m_samples, n_features)
    y: label vector shape(m_samples, )
        for both train and test
    C_value: value of the C parameter
    """
    # Create a mesh grid based on the provided axes (100 x 100 resolution)
    x0s = np.linspace(min(X_train[:,0])-0.5,max(X_train[:,0])+0.5, 100)
    x1s = np.linspace(min(X_train[:,1])-0.5,max(X_train[:,1])+0.5, 100)
    x0, x1 = np.meshgrid(x0s,x1s) # create a mesh grid
    X_mesh = np.c_[x0.ravel(), x1.ravel()] # convert all mesh points into 2-D points
    y_pred = clf.predict(X_mesh).reshape(x0.shape) # predict then covert back to the 2-D
    y_decision = clf.decision_function(X_mesh).reshape(x0.shape)

    plt.figsize=(16, 9)
    # plot the training set
    plt.plot(X_train[:, 0][y_train==0], X_train[:, 1][y_train==0], "bo", label="Class 0")
    plt.plot(X_train[:, 0][y_train==1], X_train[:, 1][y_train==1], "go", label="Class 1")

    # plot the test set
    plt.plot(X_test[:, 0][y_test==0], X_test[:, 1][y_test==0], "bx")
    plt.plot(X_test[:, 0][y_test==1], X_test[:, 1][y_test==1], "gx")

    # Plot out the support vectors (in red)
    plt.scatter(clf.support_vectors_[:,0], clf.support_vectors_[:,1], s=80, c="r", label="Support Vectors")
    # Plot decision boundary and margins
    plt.contourf(x0,x1, y_pred, cmap = plt.cm.brg, alpha = 0.1)
    plt.contourf(x0,x1, y_decision, cmap = plt.cm.brg, alpha = 0.2)
    plt.contour(x0, x1, y_decision, colors='k',
                 levels=[-1, 0, 1], alpha=0.5,
                 linestyles=['--', '-', '--'])
    plt.legend(loc="lower right")
    plt.axis("auto")

    plt.grid(True, which='both')
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("C = " + str(C_value))  # Add C value to the plot title

# Now let's plot the decision boundaries for both models
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plot_svm(best_poly_model, X_train_scaled, y_train, X_test_scaled, y_test, best_poly_model.C)
plt.title("Polynomial Kernel")

plt.subplot(1, 2, 2)
plot_svm(best_rbf_model, X_train_scaled, y_train, X_test_scaled, y_test, best_rbf_model.C)
plt.title("RBF Kernel")

plt.tight_layout()
plt.show()