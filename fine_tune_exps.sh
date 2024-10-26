## Re = 400
python main.py --config ft_NSES_NSES_5 --ntrain 5 --epochs 50 --scheduler_step 10
python main.py --config ft_NSES_NSES_5 --ntrain 25 
python main.py --config ft_NSES_NSES_5 --ntrain 100 

python main.py --config ft_NS_NSES_5 --ntrain 5 --epochs 50 --scheduler_step 10
python main.py --config ft_NS_NSES_5 --ntrain 25 
python main.py --config ft_NS_NSES_5 --ntrain 100 

python main.py --config ft_NSES_NS_5 --ntrain 5 --epochs 50 --scheduler_step 10
python main.py --config ft_NSES_NS_5 --ntrain 25 
python main.py --config ft_NSES_NS_5 --ntrain 100 

python main.py --config ft_NS_NS_5 --ntrain 5 --epochs 50 --scheduler_step 10
python main.py --config ft_NS_NS_5 --ntrain 25 
python main.py --config ft_NS_NS_5 --ntrain 100 

# ## Re = 4000
python main.py --config ft_NSES_NSES_0.5 --ntrain 5 
python main.py --config ft_NSES_NSES_0.5 --ntrain 25 
python main.py --config ft_NSES_NSES_0.5 --ntrain 100 

python main.py --config ft_NS_NSES_0.5 --ntrain 5 
python main.py --config ft_NS_NSES_0.5 --ntrain 25 
python main.py --config ft_NS_NSES_0.5 --ntrain 100 

