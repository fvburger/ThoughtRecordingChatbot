import sqlite3

sqliteConnection = None
try:
    sqliteConnection = sqlite3.connect('ANTbot.db')

    sqlite_create_table_query1 = '''CREATE TABLE CoreData (
                                participantID TEXT,
                                compDate timestamp, 
                                condition TEXT,
                                session TEXT,
                                thoughtRecord INTEGER,
                                situation TEXT,
                                emotion TEXT,
                                emoIntensity REAL,
                                emoType TEXT,
                                thoughts TEXT,
                                schemadist TEXT,
                                DATsteps INTEGER,
                                behavior TEXT,
                                PRIMARY KEY (participantID, thoughtRecord)
                                );'''

    sqlite_create_table_query2 = '''CREATE TABLE Demographics (
                                participantID TEXT PRIMARY KEY,
                                age INTEGER,
                                gender TEXT,
                                mod_score INTEGER,
                                s1comp INTEGER,
                                s1compDate timestamp,
                                s2comp INTEGER,
                                s2compDate timestamp,
                                condition TEXT,
                                modbucket INTEGER
                                );'''

    cursor = sqliteConnection.cursor()
    print("Successfully Connected to SQLite")
    cursor.execute(sqlite_create_table_query1)
    cursor.execute(sqlite_create_table_query2)
    sqliteConnection.commit()
    print("SQLite tables created")
    cursor.close()
except sqlite3.Error as error:
    print("Error while creating a sqlite table", error)
finally:
    if (sqliteConnection):
        sqliteConnection.close()
        print("sqlite connection is closed")