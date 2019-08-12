import matplotlib

def set_rc():
    matplotlib.rcdefaults()
    rcParams = {
        'font.size': 10,
        'font.family': u'sans-serif',
        'font.sans-serif': ['Arial'],
        'figure.dpi' : 300,
        'figure.autolayout': True,
    }
    matplotlib.rcParams.update(rcParams)
    
def set_size(width, aspect_ratio):
    height = width * aspect_ratio 
    assert width >= 2.63
    assert width <= 7.5
    assert height <= 8.75
    matplotlib.rcParams['figure.figsize'] = (width, height)