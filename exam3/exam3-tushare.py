import tushare as ts

# 个人token
# ts.set_token('7cb6ebc6b67bc4757d18b217c149110ad8f2654766fef3b0a18828ee')
# pro = ts.pro_api()

# 咸鱼token
pro = ts.pro_api('1a312d7a80c6fcc1fd0a28116f8a1988b6756189d4db76e5bc603031')

# 上海梅林 600073.SH
df = pro.daily(ts_code='600073.SH', start_date='2001-01-01', end_date='2022-12-31')
print(df)

# 存为csv文件
df.to_csv('600073(2).csv')
