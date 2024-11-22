string = r"""c:\Users\0xc00\Documents\RSM430\price_predict.py:220: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load('price_prediction_model.pth'))
PERSONAL INCOME FIGURES RELEASED - Statistics Canada released personal income figures yesterday, showing an increase of 0.5% in average weekly earnings. This is accompanied by a monthly increase in retail spending of 0.6%.  
Similar news found: PERSONAL INCOME FIGURES RELEASED - Statistics Canada released personal income figures yesterday, showing an increase of 0.5% in average weekly earnings. This is accompanied by a monthly increase in retail spending of 0.6%.
RNN:
   CorpBondA  CorpBondB  CorpBondC  GovtBondY2  GovtBondY5  GovtBondY10
0  -0.079935   0.030047  -0.021054   -0.004695   -0.008876    -0.010844
Max column: CorpBondB, value: 0.030046848580241203
Min column: CorpBondA, value: -0.0799347311258316
Historical Average:
    CorpBondA  CorpBondB  CorpBondC  GovtBondY2  GovtBondY5  GovtBondY10
34  -0.001308  -0.001708  -0.001356    0.000396   -0.000204     0.000331
Max column: GovtBondY2, value: 0.0003960003960004
Min column: CorpBondB, value: -0.0017078297425112


CANADIAN NON-FARM PAYROLL AND LABOR FORCE PARTICIPATION RATE - Canadian Non-farm payrolls rose by a consensus-topping 32800, while household survey saw a re-entry of workers into the labor force, with the participation rate up a tick to 63.5%
Similar news found: CANADIAN NON-FARM PAYROLL AND LABOR FORCE PARTICIPATION RATE - Canadian Non-farm payrolls rose by a consensus-topping 32800, while household survey saw a re-entry of workers into the labor force, with the participation rate up a tick to 63.5%
RNN:
   CorpBondA  CorpBondB  CorpBondC  GovtBondY2  GovtBondY5  GovtBondY10
0  -0.008059    0.00301  -0.002106    0.004902    0.009063     0.005277
Max column: GovtBondY5, value: 0.00906310877251625
Min column: CorpBondA, value: -0.00805965140104294
Historical Average:
   CorpBondA  CorpBondB  CorpBondC  GovtBondY2  GovtBondY5  GovtBondY10
4   0.003984  -0.000274   0.000712    0.001704    0.003365      0.00385
Max column: CorpBondA, value: 0.0039840637450199
Min column: CorpBondB, value: -0.0002743107941286






















"""
print(string)