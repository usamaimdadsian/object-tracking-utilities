import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 
from sklearn.linear_model import RANSACRegressor
from sklearn.gaussian_process import GaussianProcessRegressor

class PolynomialRegression(object):
    def __init__(self, degree=3, coeffs=None):
        self.degree = degree
        self.coeffs = coeffs

    def fit(self, X, y, residual_threshold=5.0):
        self.X, self.y = X,y
        self. residual_threshold = residual_threshold
        self.coeffs = np.polyfit(X.ravel(), y, self.degree)
        
    def get_params(self, deep=False):
        return {'coeffs': self.coeffs}

    def set_params(self, coeffs=None, random_state=None):
        self.coeffs = coeffs

    def predict(self, X):
        poly_eqn = np.poly1d(self.coeffs)
        y_hat = poly_eqn(X.ravel())
        
        self.residuals = self.y - y_hat
        self.inlier_mask_ = np.abs(self.residuals) < self.residual_threshold

        return y_hat

    def score(self, X, y):
        return mean_squared_error(y, self.predict(X))

        


class TrackingEvaluator:
    """
        Initialize regressor
        :param fitter_type: ransac, ransac_polynomial, ransac_gaussian, polynomial 
        :param poly_degree: degree of the polynomial
        :param data: numpy array with four columns x1,y1,w,h here x1,y1 represent top left corner
                     w and h represent width and height
        :param tracking_pos: top_left, top_right, bottom_left, bottom_right, center
    """

    
    def __init__(self,data,tracking_pos="center", fitter_type="ransac_polynomial",poly_degree=2) -> None:
        if not data.shape[1] == 4:
            raise Exception("Data should contain 4 attributes x1,y1,w,h")
        
        self.data = data
        self.fitter_type = fitter_type
        self.poly_degree = poly_degree
        self.tracking_pos = tracking_pos
        
        self.set_tracking_pos()

        
        if (fitter_type == "ransac"):
            self.model_x = RANSACRegressor(random_state=0)
            self.model_y = RANSACRegressor(random_state=0)
        elif (fitter_type == "ransac_polynomial"):
            self.model_x = RANSACRegressor(PolynomialRegression(degree=poly_degree),residual_threshold=2 * np.std(self.y),min_samples=self.x.shape[0]) 
            self.model_y = RANSACRegressor(PolynomialRegression(degree=poly_degree),residual_threshold=2*np.std(self.x),min_samples=self.y.shape[0])
        elif (fitter_type == "ransac_gaussian"):
            self.model_x = RANSACRegressor(GaussianProcessRegressor(),residual_threshold=np.std(self.y), min_samples=self.x.shape[0])
            self.model_y = RANSACRegressor(GaussianProcessRegressor(),residual_threshold=np.std(self.x), min_samples=self.y.shape[0])
        elif (fitter_type == "polynomial"):
            self.model_x = PolynomialRegression(degree=poly_degree)
            self.model_y = PolynomialRegression(degree=poly_degree)
        else:
            raise Exception("fitter type can be one of these ransac, ransac_polynomial, ransac_gaussian, polynomial")

            
        
    def set_tracking_pos(self):
        def get_center(x1,y1,w,h):
            return (x1+w)/2, (y1+h)/2
        
        def get_top_left(x1,y1,w,h):
            return x1,y1
        def get_top_right(x1,y1,w,h):
            return x1+w, y1
        def get_bottom_left(x1,y1,w,h):
            return x1, y1+h 
        def get_bottom_right(x1,y1,w,h):
            return x1+w, y1+h 
        
        positions = []
        for i in range(self.data.shape[0]):
            
            if self.tracking_pos == "center":
                point = get_center(*self.data[i])
            elif self.tracking_pos == "top_left":
                point = get_top_left(*self.data[i])
            elif self.tracking_pos == "top_right":
                point = get_top_right(*self.data[i])
            elif self.tracking_pos == "bottom_left":
                point = get_bottom_left(*self.data[i])
            elif self.tracking_pos == "bottom_right":
                point = get_bottom_right(*self.data[i])
            else:
                raise Exception("tracking_pos should be center, top_left, top_right, bottom_left or bottom_right")
            positions.append(point)
        self.positions = np.array(positions)

        self.x = self.positions[:,0]
        self.y = self.positions[:,1]

    def fit(self):
        self.X = np.expand_dims(self.x, axis=1)
        self.Y = np.expand_dims(self.y, axis=1)

        self.model_x.fit(self.X,self.y)
        self.model_y.fit(self.Y,self.x)

        
    def predict(self):
        self.y_hat = self.model_x.predict(self.X)
        self.x_hat = self.model_y.predict(self.Y)

        self.x_predicted = (self.x + self.x_hat)/2
        self.y_predicted = (self.y + self.y_hat)/2
        self.predictions = np.column_stack((self.x_predicted, self.y_predicted)) 
        # self.inlier_mask = self.model.inlier_mask_
        return self.predictions 

    """
        :param err_type: mse, mae, custom
    """ 
    def calc_error(self,err_type, x_noise=None, y_noise=None):
        positions = self.positions.copy()
        if x_noise:
            positions[:,0] += np.random.normal(0, x_noise)
        if y_noise:    
            positions[:,1] += np.random.normal(0,y_noise) 
        self.noise_positions = positions

        if err_type == "mse":
            return mean_squared_error(self.predictions, positions) 
        elif err_type == "mae":
            return mean_absolute_error(self.predictions, positions) 
        elif err_type == "custom":
            return np.mean(np.linalg.norm(self.predictions - positions, axis=1))
        else:
            raise Exception("Incorrect err_type, it should mse, mae or cutom")

    def visualize(self, draw_noisy = False):
        
        plt.figure(figsize=(12, 4), dpi=150)
        plt.title("Polynomial Fitting with "+self.fitter_type)

        legend_array = []
        if draw_noisy:
            plt.plot(self.noise_positions[:,0],self.noise_positions[:,1],'mx')
            legend_array.append("Noisy points")
        plt.plot(self.x, self.y, 'bo')
        # plt.plot(self.x[self.inlier_mask], self.y[self.inlier_mask], 'go')
        plt.plot(self.x, self.y_hat,'r-')
        plt.plot(self.x_hat, self.y, 'g-')

        plt.plot(self.x_predicted, self.y_predicted,'y-')

        plt.legend(legend_array+['Points', 'Poly x constant', 'Poly y constant','Average Predicted Poly'])
        plt.show()

        
    def accuracy(self,threshold=5,x_noise=None, y_noise=None):
        # mse = mean_squared_error(self.y_hat, self.y)
        # var_y = np.var(self.y_hat)
        # nmse = 1-(mse/var_y)
        # return nmse
        # return explained_variance_score(self.y_hat, self.y)
        # return r2_score(self.y_hat,self.y)
        # distances = np.abs(self.y - self.y_hat)
        positions = self.positions.copy()
        if x_noise:
            positions[:,0] += np.random.normal(0, x_noise)
        if y_noise:    
            positions[:,1] += np.random.normal(0,y_noise) 
 
        distances = np.linalg.norm(self.predictions - positions, axis=1)
        accuracy_list = []
        for dist in distances:
            if dist > threshold:
                accuracy_list.append(0)
            else:
                accuracy_list.append(1)
        self.noise_positions = positions
        accuracy_list = np.array(accuracy_list)
        return np.mean(accuracy_list)


    