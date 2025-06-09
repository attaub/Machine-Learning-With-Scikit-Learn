# ## Stochastic Gradient Descent


theta_path_sgd = []
m = len(X_b)
np.random.seed(42)


n_epochs = 50
t0, t1 = 5, 50  # learning schedule hyperparameters


def learning_schedule(t):
    return t0 / (t + t1)


theta = np.random.randn(2, 1)  # random initialization

for epoch in range(n_epochs):
    for i in range(m):
        if epoch == 0 and i < 20:  # not shown in the book
            y_predict = X_new_b.dot(theta)  # not shown
            style = "b-" if i > 0 else "r--"  # not shown
            plt.plot(X_new, y_predict, style)  # not shown
        random_index = np.random.randint(m)
        xi = X_b[random_index : random_index + 1]
        yi = y[random_index : random_index + 1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients
        theta_path_sgd.append(theta)  # not shown

plt.plot(X, y, "b.")  # not shown
plt.xlabel("$x_1$", fontsize=18)  # not shown
plt.ylabel("$y$", rotation=0, fontsize=18)  # not shown
plt.axis([0, 2, 0, 15])  # not shown
save_fig("sgd_plot")  # not shown
plt.show()  # not shown


theta


from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(
    max_iter=1000, tol=1e-3, penalty=None, eta0=0.1, random_state=42
)
sgd_reg.fit(X, y.ravel())


sgd_reg.intercept_, sgd_reg.coef_
