import pathlib


def get_data_path():
    path = str(pathlib.Path(__file__).parent.absolute()) + '/'
    return path


if __name__ == '__main__':
    p = get_data_path()
    print(p)
