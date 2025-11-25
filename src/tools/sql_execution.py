import sqlite3
import threading
import queue
from tabulate import tabulate
from typing import List, Tuple, Dict, Any


class SQLExecution:
    """
    A class for executing SQL queries on a SQLite database.

    Attributes:
        sql (str): The SQL query to execute.
        db_path (str): The path to the SQLite database file.
        timeout (float): The timeout for the query execution.
        success (bool): Whether the query execution was successful.
        error (Exception): The error that occurred during the query execution.
        result (List[Tuple]): The result of the query execution.
        result_cols (List[str]): The column names of the result.
        result_table (str): The result of the query execution in a table format.
    """

    def __init__(self, sql: str, db_path: str, timeout: float = 10):
        """
        Initialize the SQLExecution class.

        Args:
            sql (str): The SQL query to execute.
            db_path (str): The path to the SQLite database file.
            timeout (float): The timeout for the query execution.
        """
        self.sql = sql
        self.db_path = db_path
        self.timeout = timeout
        self.success = False
        self.error = None
        self.result = None
        
        self.result_cols = None
        self.result_table = None
        
    def _get_result_table(self, result: List[Tuple], result_cols: List[str], row_limit: int = 3, val_len_limit: int = 50):
        """
        Format the result of the query execution into a table string.

        Args:
            result (List[Tuple]): The result of the query execution.
            result_cols (List[str]): The column names of the result.
            row_limit (int): The maximum number of rows to display in the table.
            val_len_limit (int): The maximum length of the values to display in the table.

        Returns:
            str: The result of the query execution in a table format.
        """
        # if the result_cols has repeated values, add a suffix to the column name
        if len(result_cols) != len(set(result_cols)):
            result_cols = [col + "_" + str(i) for i, col in enumerate(result_cols)]
        formatted_table_rows = []
        for row in result:
            formatted_row = []
            for val in row:
                if isinstance(val, str) and len(val) > val_len_limit:
                    formatted_row.append(val[:val_len_limit] + "...")
                else:
                    formatted_row.append(val)
            formatted_table_rows.append(formatted_row)
        result_table = tabulate(formatted_table_rows, headers=result_cols, tablefmt="psql")
        return str(result_table)

    def _execute_query(self, result_queue):
        """
        Execute the SQL query and store the result in the result queue.

        Args:
            result_queue (queue.Queue): The queue to store the result of the query execution.
        """
        try:
            db_path = f'file:{self.db_path}?mode=ro'
            with sqlite3.connect(db_path, uri=True) as conn:
                conn.text_factory = lambda x: str(x, 'utf-8', errors='replace')
                cursor = conn.cursor()
                cursor.execute(self.sql)
                result = cursor.fetchall()
                result_cols = [description[0] for description in cursor.description]
                result_queue.put(("success", result, result_cols, None))
        except Exception as e:
            result_queue.put(("error", None, None, e))

    def execute(self):
        """
        Execute the SQL query and store the result in the class attributes.
        """
        result_queue = queue.Queue()
        query_thread = threading.Thread(target=self._execute_query, args=(result_queue,))
        query_thread.daemon = True
        query_thread.start()
        
        try:
            status, result, result_cols, error = result_queue.get(timeout=self.timeout)
            if status == "success":
                self.success = True
                self.error = None
                self.result = result
                self.result_cols = result_cols
                self.result_table = self._get_result_table(result, result_cols)
            else:
                self.success = False
                self.error = error
                self.result = None
                self.result_cols = None
                self.result_table = None
        except queue.Empty:
            self.success = False
            self.error = TimeoutError(f"Query execution timed out after {self.timeout} seconds")
            self.result = None
            self.result_cols = None
            self.result_table = None
            # Force the connection to close in the thread
            query_thread.join(timeout=0.1)

def execute_sql(sql: str, db_path: str, timeout: float = 10) -> Dict[str, Any]:
    """
    Execute the SQL query and return the result in a dictionary.

    Args:
        sql (str): The SQL query to execute.
        db_path (str): The path to the SQLite database file.
        timeout (float): The timeout for the query execution.

    Returns:
        Dict[str, Any]: A dictionary containing the result of the query execution.
    """
    sql_execution = SQLExecution(sql, db_path, timeout)
    sql_execution.execute()
    return {
        "success": sql_execution.success,
        "error": sql_execution.error,
        "result": sql_execution.result,
        "result_cols": sql_execution.result_cols,
        "result_table": sql_execution.result_table
    }

if __name__ == "__main__":
    result = execute_sql(
        sql="SELECT * FROM customers LIMIT 3",
        db_path="data/bird/dev/dev_databases/debit_card_specializing/debit_card_specializing.sqlite",
        timeout=30
    )
    print(f"Success: {result['success']}")
    print(f"Error: {result['error']}")
    print(f"Result: {result['result']}")
    print(f"Result cols: {result['result_cols']}")
    print(f"Result table:\n{result['result_table']}")