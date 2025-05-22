# Fast-Quiet-STaR

Doneload dataset 
```python
python download_dataset.py
```
Training and eval Fast Quiet-STaR
```bash
bash run_sh\fast_quiet_star_mistral.sh
```
Make Fast Quiet-STaR NTP Training data
```bash
bash run_sh\infer_web_math.sh $output_path $checkpoint $n_ahead
```
Training and eval Fast Quiet-STaR NTP
```bash
bash run_sh\mistral_last.sh
```
