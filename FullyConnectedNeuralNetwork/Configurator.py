import configparser
from Projects.Project1.DataGeneration import DataGeneration
from Projects.Project1.Network import Network


config = configparser.ConfigParser()

#config.read('demo1.txt')
config.read('demo2.txt')
#config.read('demo3.txt')
#config.read('demo4.txt')

layerspecs = []
globals  = {}
for section in config.sections():
    if section == 'GLOBALS':  # Creating a dictionary that is passed to the Network class constructor
        for key in config['GLOBALS']:
            if key == 'verbose':
                globals['verbose'] = config['GLOBALS'].getboolean('verbose')
            elif key == 'batch_size':
                globals[key] = config['GLOBALS'].getint(key)
            else:
                try:
                    globals[key] = float(config['GLOBALS'][key])
                except ValueError:
                    globals[key] = config['GLOBALS'][key]
    elif section == 'DATA':
        datadict = {}
        for key in config['DATA']:
            if key == 'fig_centered' or key == 'draw' or key == 'flatten':
                datadict[key] = config['DATA'].getboolean(key)
            elif key == 'train_val_test':
                datadict[key] = tuple(map(float, config['DATA'].get('train_val_test').strip('(').strip(')').split(', ')))
            elif key == 'img_size':
                datadict[key] = int(config['DATA'][key])
            elif key == 'set_size':
                datadict[key] = int(config['DATA'][key])
            else:
                try:
                    datadict[key] = float(config['DATA'][key])
                except ValueError:
                    datadict[key] = config['DATA'][key]
    else:  # Creating the collection of dictionaries that is passed to Networks gen_network method.
        layerdict = {}
        for key in config[section]:
            if key == 'w_range':
                layerdict[key] = tuple(map(float, config[section].get('w_range').strip('(').strip(')').split(', ')))
            else:
                try:
                    layerdict[key] = float(config[section][key])
                except ValueError:
                    layerdict[key] = config[section][key]
        layerspecs.append(layerdict)

print(datadict)
print(layerspecs)
new_network = Network(layerspecs, globals['loss'], globals['verbose'], globals)
new_network.gen_network()


data = DataGeneration(noise=datadict['noise'], img_size=datadict['img_size'], set_size=datadict['set_size'],
                      flatten=datadict['flatten'], fig_centered=datadict['fig_centered'],
                      train_val_test=datadict['train_val_test'], draw=datadict['draw'])
data.gen_dataset()

new_network.train(data.train_set, data.val_set, data.test_set, globals['batch_size'])