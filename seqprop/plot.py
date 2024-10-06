import numpy as np
import pandas as pd
import logomaker
import matplotlib.pyplot as plt

def pwm_logo(pwm: np.array, figsize=(16, 2)):
    # create matplotlib axis
    fig, ax = plt.subplots(figsize=figsize)

    logomaker.Logo(
        pd.DataFrame(pwm, columns=list('ACGT')), 
        shade_below=.5,
        fade_below=.5,
        ax=ax
    )

    return fig