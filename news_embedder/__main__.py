import sys

if __name__ == '__main__':
    command = sys.argv[1]
    if command == 'land':
        from news_embedder.cli.land import land
        wd = sys.argv[2]
        td = './settings'
        land(wd)
