import os
import os.path

LOG_ROOT_PATH = 'logs'

for x in os.listdir(LOG_ROOT_PATH):
    os.remove(os.path.join(LOG_ROOT_PATH, x))
