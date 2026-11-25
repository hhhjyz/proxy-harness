from pathlib import Path
import sys
sys.path.insert(0, '/workspace/scratch/bbcc99ac5c90/proxy-harness-v1')
from proxy_harness.__main__ import main
main(['render-check','--scene',str(Path(__file__).with_name('scene.html')),'--spec',str(Path(__file__).with_name('spec.json')),'--out',str(Path(__file__).with_name('local_check'))] + sys.argv[1:])
