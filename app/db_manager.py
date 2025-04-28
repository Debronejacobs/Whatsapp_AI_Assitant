# whatsapp_client/app/db_manager.py
import os
import urllib.parse
import aiosqlite
import logging
from typing import Optional, List, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConversationDBManager:
    """
    Manages asynchronous interaction with the SQLite database for conversation history.
    """
    def __init__(self, db_url: str):
        """
        Initializes the database manager.

        Args:
            db_url: The database URL string (e.g., "sqlite:///path/to/db.sqlite").
        """
        parsed_url = urllib.parse.urlparse(db_url)
        if parsed_url.scheme != 'sqlite':
            raise ValueError(f"Invalid database URL scheme: {parsed_url.scheme}. Only 'sqlite' is supported.")

        # Ensure the path is absolute
        self.db_path = os.path.abspath(parsed_url.path)
        self._connection: Optional[aiosqlite.Connection] = None
        logger.info(f"Initialized ConversationDBManager for DB path: {self.db_path}")

    async def connect(self):
        """
        Establishes the database connection and creates the table if needed.
        """
        if self._connection is not None:
            logger.warning("Database connection already established.")
            return

        logger.info(f"Connecting to database: {self.db_path}")
        try:
            db_dir = os.path.dirname(self.db_path)
            if db_dir: os.makedirs(db_dir, exist_ok=True) # Ensure directory exists

            self._connection = await aiosqlite.connect(self.db_path)
            logger.info("Database connection established successfully.")
            await self._create_messages_table()
            logger.info("Ensured 'messages' table exists.")
        except Exception as e:
            logger.critical(f"Failed to connect to database or create table: {e}", exc_info=True)
            if self._connection is not None: await self._connection.close()
            self._connection = None
            raise

    async def close(self):
        """ Closes the database connection. """
        if self._connection is not None:
            logger.info("Closing database connection.")
            await self._connection.close()
            self._connection = None
            logger.info("Database connection closed.")
        else:
            logger.warning("Attempted to close database connection, but no connection was open.")

    async def _create_messages_table(self):
        """ Creates the 'messages' table if it does not already exist. """
        if self._connection is None: raise ConnectionError("Database not connected.")
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender_phone TEXT NOT NULL,
            role TEXT NOT NULL,
            message_id TEXT,
            type TEXT NOT NULL,
            text_content TEXT,
            local_file_path TEXT,
            timestamp INTEGER NOT NULL
        );
        """
        # Add index for faster history lookup
        create_index_sql = """
        CREATE INDEX IF NOT EXISTS idx_sender_phone_timestamp ON messages (sender_phone, timestamp);
        """
        async with self._connection.cursor() as cursor:
            await cursor.execute(create_table_sql)
            await cursor.execute(create_index_sql)
        await self._connection.commit()

    async def save_message(
        self,
        sender_phone: str,
        role: str,
        message_id: Optional[str],
        type: str,
        text_content: Optional[str],
        local_file_path: Optional[str],
        timestamp: int
    ):
        """ Saves a conversation message to the database. """
        if self._connection is None: raise ConnectionError("Database not connected.")
        insert_sql = """
        INSERT INTO messages (sender_phone, role, message_id, type, text_content, local_file_path, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """
        values = (sender_phone, role, message_id, type, text_content, local_file_path, timestamp)
        try:
            async with self._connection.cursor() as cursor:
                await cursor.execute(insert_sql, values)
            await self._connection.commit()
            logger.debug(f"Saved message for {sender_phone} (role: {role}, type: {type}).")
        except Exception as e:
            logger.error(f"Failed to save message for {sender_phone}: {e}", exc_info=True)
            raise

    async def get_history(self, sender_phone: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Retrieves recent conversation history for a given phone number, oldest first.
        """
        if self._connection is None: raise ConnectionError("Database not connected.")
        # Get the most recent `limit` messages, then sort ascending (oldest first)
        # Using a subquery is generally efficient for getting the latest N then sorting
        select_sql = """
        SELECT id, sender_phone, role, message_id, type, text_content, local_file_path, timestamp
        FROM (
            SELECT * FROM messages
            WHERE sender_phone = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ) sub
        ORDER BY timestamp ASC;
        """
        try:
            async with self._connection.cursor() as cursor:
                await cursor.execute(select_sql, (sender_phone, limit))
                rows = await cursor.fetchall()
                if not rows: return [] # Return empty list if no history
                column_names = [description[0] for description in cursor.description]

            history = [dict(zip(column_names, row)) for row in rows]
            logger.debug(f"Retrieved {len(history)} history messages for {sender_phone}.")
            return history
        except Exception as e:
            logger.error(f"Failed to retrieve history for {sender_phone}: {e}", exc_info=True)
            raise