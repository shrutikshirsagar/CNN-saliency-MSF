import pandas as pd
file_label = '/media/amrgaballah/Backup_Plus/stress/final_labels.csv'
output_cols = ['filename','Arousal', 'valence']
df = pd.read_csv(file_label)
print(df.shape)
dfn = df['filename']

df1 = df[['Arousal', 'valence']]
print(df1)
df1[df1 <2.5] = 0
df1[df1 >=2.5] = 1
print(dfn.shape, df1.shape)
out = np.hstack((dfn[:,None],df1))
df_final1=pd.DataFrame(out, columns=output_cols)
df_final1.to_csv('/media/amrgaballah/Backup_Plus/IEMPCAP_final_label.csv',index=None)
