import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import csv

class SalesPredictionGUI:

    def __init__(self, master):
        self.master = master
        master.title("Sales Growth Prediction")

        self.add_title_labal = tk.Label(master, text="Sales predictor", font=('Arial', 24) , bg='yellow', fg='green')
        self.add_title_labal.pack()

        # Create buttons to add data, view data, and view plots
        self.add_data_button = tk.Button(master, text="Select CSV File", command=self.add_data)
        self.add_data_button.pack()

        self.add_newData_button = tk.Button(master, text="Add Data", command=self.add_data_to_csv)
        self.add_newData_button.pack()

        self.view_data_button = tk.Button(master, text="View Data", command=self.view_data)
        self.view_data_button.pack()

        self.view_plots_button = tk.Button(master, text="View Predictions plots", command=self.view_predictions)
        self.view_plots_button.pack()


        # Initialize the DataFrame to store sales data
        self.data = pd.DataFrame(columns=['Year', 'Month', 'Sales Growth'])

    def add_data(self):
        # Open a file dialog to select a CSV file with sales data
        file_path = filedialog.askopenfilename(title="Select Sales Data File", filetypes=[("CSV Files", "*.csv")])

        # Load the data into the DataFrame
        new_data = pd.read_csv(file_path)
        self.data = self.data.append(new_data, ignore_index=True)

    def view_data(self):
         # Create the table window
        table_window = tk.Toplevel(root)
        table_window.title('Sales Data')
    
        # Create the canvas for the table and scrollbar
        canvas = tk.Canvas(table_window)
        canvas.pack(side='left', fill='both', expand=True)
    
        # Create the scrollbar and attach it to the canvas
        scrollbar = tk.Scrollbar(table_window, orient='vertical', command=canvas.yview)
        scrollbar.pack(side='right', fill='y')
        canvas.configure(yscrollcommand=scrollbar.set)
    
        # Create the table as a frame inside the canvas
        table = tk.Frame(canvas)
        table.pack(side='top', fill='both', expand=True)
        canvas.create_window((0, 0), window=table, anchor='nw')
    
        # Create the table headers
        headers = ['Year', 'Month', 'Sales Growth']
        for i, header in enumerate(headers):
            label = tk.Label(table, text=header, relief=tk.RIDGE, width=15)
            label.grid(row=0, column=i)
    
         # Fill in the table data
        for i, row in self.data.iterrows():
            year = tk.Label(table, text=row['Year'], relief=tk.RIDGE, width=15)
            year.grid(row=i+1, column=0)
            month = tk.Label(table, text=row['Month'], relief=tk.RIDGE, width=15)
            month.grid(row=i+1, column=1)
            sales = tk.Label(table, text=row['Sales Growth'], relief=tk.RIDGE, width=15)
            sales.grid(row=i+1, column=2)
    
        # Configure the canvas to expand when the table is resized
        table.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))


    def view_predictions(self):
        # Split the data into training and testing sets
        

        self.data['Date'] = pd.to_datetime(self.data[['Year', 'Month']].assign(Day=1))

        # Split the data into training and testing sets
        train_data = self.data[self.data['Year'] < 2022]
        test_data = self.data[self.data['Year'] == 2022]

        # Prepare the data for training
        X_train = train_data[['Year', 'Month']]
        y_train = train_data['Sales Growth']

        # Train a linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Prepare the data for testing
        X_test = test_data[['Year', 'Month']]
        y_test = test_data['Sales Growth']

        # Make predictions on the testing data
        y_pred = model.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        print('Root Mean Squared Error: ', rmse)

        # Plot the training data and model predictions
        plt.figure(figsize=(11, 6))
        plt.plot(train_data['Date'], train_data['Sales Growth'], label='Training Data')
        plt.plot(test_data['Date'], test_data['Sales Growth'], label='Testing Data')
        plt.plot(test_data['Date'], y_pred, label='Predictions')
        plt.xlabel('Date')
        plt.ylabel('Sales Growth')
        plt.title('Sales Growth Prediction')
        plt.legend()
        plt.show()


        # Plot the residuals
        residuals = y_test - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(X_test['Month'], residuals)
        plt.xlabel('Month')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.show()

        # Check the normality of the residuals
        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=10)
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Normality Check')
        plt.show()

        # Check the linearity assumption
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred)
        plt.xlabel('Actual Values')
        plt.ylabel('Predictions')
        plt.title('Linearity Check')
        plt.show()

        # Check the homoscedasticity assumption
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals)
        plt.xlabel('Predictions')
        plt.ylabel('Residuals')
        plt.title('Homoscedasticity Check')
        plt.show()
     
    def add_data_to_csv(self):

      # Create the form window
        form_window = tk.Toplevel(root)
        form_window.title('Add Data')

        # Create the form fields
        year_label = tk.Label(form_window, text='Year')
        year_label.grid(row=0, column=0)
        year_entry = tk.Entry(form_window)
        year_entry.grid(row=0, column=1)

        month_label = tk.Label(form_window, text='Month')
        month_label.grid(row=1, column=0)
        month_entry = tk.Entry(form_window)
        month_entry.grid(row=1, column=1)

        sales_label = tk.Label(form_window, text='Sales Growth')
        sales_label.grid(row=2, column=0)
        sales_entry = tk.Entry(form_window)
        sales_entry.grid(row=2, column=1)

        # Create the add data button
        add_button = tk.Button(form_window, text='Add Data', command=lambda: add_data(year_entry.get(), month_entry.get(), sales_entry.get()))
        add_button.grid(row=3, column=1)

        def add_data(year, month, sales):
             
             with open('sales_data.csv', mode='r') as file:
                reader = csv.reader('tilak_di_hatti_sales.csv')
                for row in reader:
                    if row[0] == year and row[1] == month :
                    # Data already exists, show error message and return
                        tk.messagebox.showerror('Error', 'Data already exists in the file.')
                        return
        # Append the data to the CSV file
             with open('tilak_di_hatti_sales.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([year, month, sales])
                tk.messagebox.showinfo('Success', 'Data added successfully.')
         # Clear the form fields
        
        year_entry.delete(0, tk.END)
        month_entry.delete(0, tk.END)
        sales_entry.delete(0, tk.END)


    

# Create the GUI
root = tk.Tk()
root.minsize(400, 200)
root.maxsize(1200, 900)
root.configure(bg='lightblue')
gui = SalesPredictionGUI(root)
root.mainloop()
