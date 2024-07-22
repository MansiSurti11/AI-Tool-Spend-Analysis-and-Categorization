import psycopg2  # Importing psycopg2 to connect to PostgreSQL databases
import pandas as pd  # Importing pandas for data manipulation and analysis 

# Function to connect to PostgreSQL and insert data
def insert_into_postgresql(data, table_name, host='localhost', port='5432', database='supplier', user='postgres', password='mansi123'):
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(host=host, port=port, database=database, user=user, password=password)  # Establishing connection to the PostgreSQL database
        cursor = conn.cursor()  # Creating a cursor object to execute SQL queries

        # Create insert query
        insert_query = f"""
        INSERT INTO {table_name} (
            age, gender, item_purchased, category, purchase_amount_usd, location, size, color, season, review_rating, 
            subscription_status, shipping_type, discount_applied, promo_code_used, previous_purchases, 
            payment_method, frequency_of_purchases
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        # Formatting the SQL insert query with table name and column names

        # Execute the insert query for each row in the data
        for row in data:
            cursor.execute(insert_query, row)  # Executing the insert query for each row in the provided data

        conn.commit()  # Committing the transaction to the database
        print(f"Data inserted successfully into PostgreSQL table '{table_name}'!")

    except (Exception, psycopg2.Error) as error:
        # Catching any exception that occurs during the database operation
        print("Error while connecting to PostgreSQL or inserting data:", error)

    finally:
        # Closing database connection
        if conn:
            cursor.close()  # Closing the cursor
            conn.close()  # Closing the connection to the database
            print("PostgreSQL connection is closed")

# Example usage:
if __name__ == "__main__":
    # Example data (list of tuples)
    data = [
        (30, 'Male', 'Shoes', 'Footwear', 100, 'New York', 'Large', 'Black', 'Summer', 4.5, 'Active', 'Express', '20OFF', 'ABC123', 5, 'Credit Card', 'Monthly'),
        # Add more tuples as needed
    ]

    table_name = 'spend_data'  # Replace with your table name
    insert_into_postgresql(data, table_name, host='localhost', port='5432', database='supplier', user='postgres', password='mansi123')
    # Calling the function to insert data into the specified PostgreSQL table
