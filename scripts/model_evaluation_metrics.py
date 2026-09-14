import numpy as np
from sklearn.metrics import root_mean_squared_error

def minMaxRMSE(
       rmse,
       y_test
    ):

    target_range = (y_test.max() - y_test.min())
    
    nrmse_range = (rmse / target_range)
    
    print(f"RMSE to MinMax: {nrmse_range:.2%}")

def stdRMSE(
       rmse,
       y_test
    ):

    std = np.std(y_test)
    
    relative_rmse_std = rmse / std
    
    print(f"RMSE to STD: {relative_rmse_std}")

def RMSE(
        y_test,
        y_pred
    ):

    rmse = root_mean_squared_error(
        y_test,
        y_pred
    )

    print(f"RMSE: {rmse}")

    return rmse
    
    