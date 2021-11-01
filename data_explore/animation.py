import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
from common import get_data_path
from get_data.static import RESTfulProcessor
import seaborn as sns


class AnimatedScatter:
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    save_folder = get_data_path() + 'data_explore/'

    def __init__(self, df, filename: str, frame_column: str, x_col: str, y_col: str, c_col: str, axis):
        self.data = df

        self.save_path = self.save_folder + filename

        self.frame_column = frame_column
        self.frames = np.sort(list(df[frame_column].unique()))
        self._frame_id = 0  # start from 1 because all values are missing in the first date
        self.handles = []

        self.fig, self.ax = plt.subplots()
        self.ax.axis(axis)
        self.x_col = x_col
        self.y_col = y_col
        self.c_col = c_col

        self.colors = sns.color_palette('Paired', n_colors=self.data[self.c_col].nunique())
        self.ani = None
        self.scat = None
        self.prev_scat = None

    def get_frame_id(self):
        frame_id = self._frame_id
        self._frame_id += 1

        return frame_id

    def update(self, i):
        """Update the scatter plot."""
        frame_id = self.get_frame_id()
        print('start {}th frame'.format(frame_id))

        new_alphas = []
        for s in self.handles:
            new_alpha = s.get_alpha() * 0.6
            new_alphas.append(new_alpha)
            if new_alpha > 0.1:
                s.set_alpha(new_alpha)
                s.set_sizes([15])
            else:
                s.remove()

        if len(self.handles) > 0:
            n = len(new_alphas)
            self.handles = [self.handles[i] for i in range(n) if new_alphas[i] > 0.1]

        self.ax.set_title('{}'.format(self.frames[frame_id]))

        df_tmp_all = self.data[self.data[self.frame_column] == self.frames[frame_id]].copy()
        grouped = df_tmp_all.groupby([self.c_col])
        dict_scat = {}
        for l, df_tmp in grouped:
            x = df_tmp[self.x_col].values
            y = df_tmp[self.y_col].values
            c = pd.Series(self.colors, index=df_tmp_all[self.c_col])
            scat = self.ax.scatter(x, y, color=c[l], label=l, alpha=1)
            dict_scat[l] = scat
            self.handles.append(scat)

        self.ax.legend(dict_scat.values(), dict_scat.keys())

        print('{}th frame is done'.format(frame_id))

    def run(self):
        self.ani = animation.FuncAnimation(
            self.fig,
            self.update,
            frames=len(self.frames) - 1,  # remove the frame in setup phase
            interval=500,  # change frame every 0.5 seconds
            repeat=False
        )
        self.ani.save(self.save_path, writer='imagemagick')


if __name__ == '__main__':
    resp = RESTfulProcessor(verbose=False)
    df_train, _ = resp.process_data()

    tickers = ['AAPL', 'MSFT', 'AMZN', 'FB', 'GOOG', 'TSLA', 'NVDA']
    df_sample = df_train.loc[df_train['ticker'].isin(tickers), ['ticker', 'date', 'open_pct_chg', 'volume_pct_chg']]
    df_sample['ticker'] = pd.Categorical(df_sample['ticker'])
    df_sample.dropna(inplace=True)
    ani_scat = AnimatedScatter(
        df=df_sample,
        filename='test.gif',
        frame_column='date',
        x_col='volume_pct_chg',
        y_col='open_pct_chg',
        c_col='ticker',
        axis=[-50, 50, -10, 10]
    )
    ani_scat.run()
