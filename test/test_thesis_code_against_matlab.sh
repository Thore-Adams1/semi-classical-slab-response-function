rm ./.validation -r
echo "========================== Testing single_cpu mode ====================="
python thesis_code.py A2 A1 H G -p lc=8 steps=20 P=0 -v Kx=0.0002,0.001 -v w=0.5622,0.8 L=100,10000 tau=100,200 -wo debug/output_single_cpu.pkl -f
python validate_pickle.py debug/output_single_cpu.pkl
echo "========================== Testing multi_cpu mode ======================"
python thesis_code.py A2 A1 H G -p lc=8 steps=20 P=0 -v Kx=0.0002,0.001 -v w=0.5622,0.8 L=100,10000 tau=100,200 -wo debug/output_multi_cpu.pkl -fx
python validate_pickle.py debug/output_multi_cpu.pkl
echo "========================== Testing gpu mode ============================"
python thesis_code.py A2 A1 H G -p lc=8 steps=20 P=0 -v Kx=0.0002,0.001 -v w=0.5622,0.8 L=100,10000 tau=100,200 -wo debug/output_gpu.pkl -f --gpu
python validate_pickle.py debug/output_gpu.pkl
