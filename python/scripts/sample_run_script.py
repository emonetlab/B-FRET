import sys
sys.path.append('../src')
from load_save import load_data
from main import analyze_FRET_data

DATA_DIR = r'../example'
EXP_DIR = r'../example/gauss_noise/sinusoids'
	
def run():

	data = load_data(DATA_DIR, 'data_sinusoids')
	a = analyze_FRET_data(data, EXP_DIR)
	a.run()

if __name__ == '__main__':
	run()