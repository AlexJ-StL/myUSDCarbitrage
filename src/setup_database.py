import os
import psycopg2

# We'll use environment variables if available, but provide defaults
# No dotenv dependency required


def setup_database():
    """
    Set up the database using the SQL commands from database_setup.sql
    """
    # Connect to default postgres database first (to create our new database)
    conn = psycopg2.connect(
        dbname="postgres",
        user=os.environ.get("POSTGRES_USER", "postgres"),
        password=os.environ.get("POSTGRES_PASSWORD", ""),
        host=os.environ.get("DB_HOST", "localhost"),
    )
    conn.autocommit = True  # Required for CREATE DATABASE

    # Read SQL file
    sql_file_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), ".sql", "database_setup.sql"
    )
    with open(sql_file_path, "r") as f:
        sql_script = f.read()

    # Split the script into individual commands
    # This simple approach works for basic SQL but might need refinement for complex scripts
    commands = sql_script.split(";")

    try:
        with conn.cursor() as cur:
            # Execute each command separately
            for command in commands:
                command = command.strip()
                if command:  # Skip empty commands
                    print(f"Executing: {command[:60]}...")
                    try:
                        cur.execute(command)
                        print("Success!")
                    except psycopg2.Error as e:
                        print(f"Error executing command: {e}")
                        # Continue with other commands even if one fails
    except Exception as e:
        print(f"Database setup error: {e}")
    finally:
        conn.close()
        print("Database setup completed.")


if __name__ == "__main__":
    setup_database()
