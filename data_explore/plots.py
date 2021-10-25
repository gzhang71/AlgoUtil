import pandas as pd
from get_data.static import RESTfulProcessor

pd.options.display.width = 0

resp = RESTfulProcessor(verbose=False)
df_train, df_test = resp.process_data()
