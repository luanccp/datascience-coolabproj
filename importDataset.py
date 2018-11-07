import pandas as pd
file = pd.ExcelFile('test.xlsx')
# print(type(file))

file.sheet_names
# dois modos de imprime a tabela inteira
df1 = file.parse('Planilha1')
df1 = file.parse(0)

print(df1)
