mall_query = '''
             SELECT *
             FROM customers
             '''

mall_database = 'mall_customers'

mall_url = get_db_url(host, user, password, mall_database)

mall_df = pd.read_sql(mall_query, mall_url)

mall_df.head()