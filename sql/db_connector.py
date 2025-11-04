import mysql.connector
from mysql.connector import Error
from loguru import logger

DB_CONFIG = {
    'host': 'localhost',
    'database': 'text_to_model_db',
    'user': 'admin',
    'password': 'admin'
}

class DatabaseConnector:
    def __init__(self, config: dict = DB_CONFIG):
        self.config = config
        self.connection = None

    def __enter__(self):
        try:
            self.connection = mysql.connector.connect(**self.config)
            if self.connection.is_connected():
                logger.info("[DB] Connection established.")
                return self
        except Error as e:
            logger.error(f"[DB] Error connecting to MySQL: {e}")
            raise RuntimeError(f"Database connection error: {e}") from e

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("[DB] Connection closed.")
        return False

    def execute_query(self, query: str, params: tuple = None, fetch_one: bool = False):
        cursor = None
        is_select_query = query.strip().upper().startswith('SELECT')
        
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(query, params)

            if is_select_query:
                if fetch_one:
                    result = cursor.fetchone()
                else:
                    result = cursor.fetchall()
                return result
            else:                
                self.connection.commit()
                return cursor.lastrowid if cursor.lastrowid else None
                
        except Error as e:
            logger.error(f"[DB] Error executing query: {e}")
            if not is_select_query and self.connection:
                try:                    
                    self.connection.rollback()
                    logger.warning("[DB] Transaction rolled back.")
                except Error as re:
                    logger.error(f"[DB] Error during rollback: {re}")
            raise 
        finally:
            if cursor:
                cursor.close()

    def save_feedback(
        self,
        is_positive: int,
        original_prompt: str,
        refined_prompt: str = None,
        chat: str = None,
        negative_prompt: str = None,
        video_frame_url: str = None,
        step: str = None,
        negative_feedback: str = None
    ):
        """
        Salva um novo registro de feedback no banco de dados.
        """
        query = """
            INSERT INTO feedback (
                is_positive, original_prompt, refined_prompt, chat, 
                negative_prompt, video_frame_url, step, negative_feedback
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        return self.execute_query(
            query,
            params=(
                is_positive,
                original_prompt,
                refined_prompt,
                chat,
                negative_prompt,
                video_frame_url,
                step,
                negative_feedback
            )
        )

    def save_multiview(self, feedback_id, image_url):
        query = """
            INSERT INTO multiviews (feedback_id, image_url)
            VALUES (%s, %s)
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, (feedback_id, image_url))
            self.connection.commit()
            cursor.close()
        except Exception as e:
            logger.error(f"[DB] Failed to save multiview: {e}")
            raise

    def get_multiview_paths(self, feedback_id: int):
        """Get all multiview file paths for a given feedback ID."""        
        query = "SELECT image_url FROM multiviews WHERE feedback_id = %s"
        results = self.execute_query(query, params=(feedback_id,))        
        return [row['image_url'] for row in results] if results else []

    def delete_multiviews_by_feedback_id(self, feedback_id: int):
        """Delete all multiviews associated with a feedback ID."""
        query = "DELETE FROM multiviews WHERE feedback_id = %s"
        self.execute_query(query, params=(feedback_id,))

    def delete_feedback(self, feedback_id: int):
        """Delete a feedback entry."""
        query = "DELETE FROM feedback WHERE id = %s"        
        self.execute_query(query, params=(feedback_id,))
        
    def get_multiview_count(self):
        """Get total count of multiview images in database."""
        query = "SELECT COUNT(*) as count FROM multiviews"        
        result = self.execute_query(query, fetch_one=True)
        return result['count'] if result else 0

    def get_oldest_feedback_with_multiviews(self):
        """Get the oldest feedback entry that has associated multiviews."""
        query = """
            SELECT f.id, COUNT(m.id) as multiview_count
            FROM feedback f
            INNER JOIN multiviews m ON f.id = m.feedback_id
            GROUP BY f.id
            ORDER BY f.created_at ASC
            LIMIT 1
        """

        return self.execute_query(query, fetch_one=True)

def get_db_connector():
    with DatabaseConnector() as connector:
        yield connector