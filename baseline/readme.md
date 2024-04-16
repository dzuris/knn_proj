## Baseline

Model configuration is in this repository subfolder `/cfg`

To run baseline:

1. Download [model repository](https://github.com/videturfortuna/vehicle_reid_itsc2023/tree/main)
2. Download [dataset](https://drive.google.com/file/d/0B0o1ZxGs_oVZWmtFdXpqTGl3WUU/view?resourcekey=0-YIcgC3HmQD7QnvoMpmfczA)
3. Update dataset path to match paths in `/cfg/config.yaml` file
4. Run ``python eval.py --path_weights <your_path>/cfg/``
5. Install dependences if needed and run again
6. Enjoy

The results from evaluation:

| mAP | CMC1 | CMC5 |
| :-: | :-: | :-: |
| 0.8614320489016402 | 0.9791418313980103 | 0.9916567206382751 |
