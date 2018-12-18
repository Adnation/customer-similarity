import os
import psycopg2
import pandas as pd
from flask import jsonify
from constants import const
from sqlalchemy import create_engine
from local_settings import db_configs
from sklearn.neighbors import NearestNeighbors
from flask import Flask, request, render_template

# Flask server app
app = Flask(__name__)

# Postgres connection and engine object
pg_connection = None
pg_engine = None


# Method to get database connection
def get_db_connection():
    global pg_connection
    global pg_engine
    # Check if global variable is connected or not
    if pg_connection is not None:
        return pg_connection, pg_engine
    # Connect with database and return connection and engine object
    else:
        pg_connection = psycopg2.connect(
            database=db_configs['database'],
            user=db_configs['user'],
            password=db_configs['password'],
            host=db_configs['host'],
            port=db_configs['port']
        )

        pg_engine = create_engine(
            'postgresql://' + db_configs['user'] +
            ':' + db_configs['password'] + '@' +
            db_configs['host'] + ':' +
            db_configs['port'] + '/' +
            db_configs['database'], echo=False
        )
        return pg_connection, pg_engine


# Method to calculate distance matrix
def prepare_distance_matrix():

    print('Distance calculation initiated')
    # Get database connection and engine object
    pg_connection, engine = get_db_connection()

    # Cursor to extract data from user_course_view table
    cursor = pg_connection.cursor()
    cursor.execute('''
        SELECT DISTINCT user_handle FROM {table_name}
    '''.format(table_name=const['tables']['TABLE_USER_COURSE_VIEWS']))

    unique_user_handles = cursor.fetchall()

    user_tag_times = []
    # Iterate over all the users
    for index, user_handle in enumerate(unique_user_handles):

        user_handle = user_handle[0]

        if index % 500 == 0:
            print(index)

        # Dict to store tag and amount of time spend on it
        tag_time_dict = {
            'user_handle': user_handle
        }

        # Get all the unique tags from DB
        cursor.execute('''
            SELECT DISTINCT course_tags FROM {table_name}
        '''.format(table_name=const['tables']['TABLE_COURSE_TAGS']))
        unique_course_tags = cursor.fetchall()

        # Iterate over course to create skelaton of tags with 0 as time spent
        for u_tag in unique_course_tags:
            u_tag = u_tag[0]
            tag_time_dict[u_tag] = 0

        # Get amount of time use spent on course
        grouped_user_course_view_df = pd.read_sql_query('''
            SELECT user_handle, course_id, SUM(view_time_seconds) AS view_time_seconds
            FROM {table_name}
            WHERE user_handle = {user_handle}
            GROUP BY 1, 2
        '''.format(
            table_name=const['tables']['user_course_views'],
            user_handle=user_handle
        ), con=engine)

        # Total time user spent on all the courses
        total_time_spent = grouped_user_course_view_df['view_time_seconds'].sum()

        # Iterate over courses
        for index, row in grouped_user_course_view_df.iterrows():
            # If user has spend less than 1 hour then skip
            if row['view_time_seconds'] < 3600:
                continue
            course = row['course_id']

            # Extract course tags for course
            cursor.execute('''
                SELECT course_tags FROM {table_name}
                WHERE course_id = '{course}'
            '''.format(
                table_name=const['tables']['TABLE_COURSE_TAGS'],
                course=course))

            course_tags = cursor.fetchall()

            # Update course tag and time
            for tag in course_tags:
                tag = tag[0]
                if tag in tag_time_dict.keys():
                    tag_time_dict[tag] += row['view_time_seconds'] / len(course_tags)
                else:
                    tag_time_dict[tag] = row['view_time_seconds'] / len(course_tags)

        # Append it to list
        user_tag_times.append(tag_time_dict)

    # Create dataframe from list of dict
    df = pd.DataFrame(user_tag_times)
    # Write results to postgres table
    df.to_sql('user_tag_time', engine, index=False)


def import_data_to_db():
    # Get database connection and engine
    pg_connection, engine = get_db_connection()

    # Check if required tables exists or not
    cursor = pg_connection.cursor()
    cursor.execute('''
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public'
    ''')

    # Find the missing tables
    existing_tables = set([i[0] for i in cursor.fetchall()])
    missing_tables = list(set(const['tables'].values()) - existing_tables)

    # Iterate over missing tables
    for missing_table in missing_tables:

        if missing_table == const['tables']['TABLE_USER_TAG_TIME']:
            continue

        print(missing_table + ' not found.')

        # Find corresponding CSV
        csv_path = os.path.join(
            const['CSV_FILE_PATH'], missing_table + '.csv'
        )

        # Raise exception if csv not found
        if not os.path.exists(csv_path):
            raise Exception('CSV data not found for data import')
            continue

        # Read CSV
        df = pd.read_csv(csv_path)
        # Drop missing value rows
        df.dropna(inplace=True)
        # Write data to postgres
        df.to_sql(missing_table, engine, index=False)
        print(missing_table + ' imported from CSV')

    # If calculation metric is missing the add that table
    if const['tables']['TABLE_USER_TAG_TIME'] in missing_tables:
        prepare_distance_matrix()


# Find courses and tags of a user
def get_courses_and_tags(user_handle):
    # Get connection objects
    pg_connection, _ = get_db_connection()
    cursor = pg_connection.cursor()

    # Get all the course id of user
    cursor.execute('''
        SELECT course_id, sum(view_time_seconds) AS view_time_seconds FROM {table_name}
        WHERE user_handle = {user_handle}
        GROUP BY user_handle, course_id
        ORDER BY view_time_seconds
    '''.format(
        table_name=const['tables']['TABLE_USER_COURSE_VIEWS'],
        user_handle=user_handle))

    # Flatten the list
    courses = cursor.fetchall()
    courses = ["'" + c[0] + "'" for c in courses]

    # Get all the tags of course
    cursor.execute('''
        SELECT DISTINCT course_tags FROM {table_name}
        WHERE course_id IN ({courses})
        '''.format(
        table_name=const['tables']['TABLE_COURSE_TAGS'],
        courses=', '.join(courses)))

    # Flatter to list
    tags = cursor.fetchall()
    tags = [t[0] for t in tags]
    courses = [c.replace("'", "") for c in courses]
    return courses, tags


def get_similar_user(user_handle):
    # Get engine object
    _, engine = get_db_connection()

    # Get entire distance matrix
    df = pd.read_sql_query(
        'SELECT * FROM {table}'.format(table=const['tables']['TABLE_USER_TAG_TIME']),
        con=engine
    )

    columns = list(df.columns)
    columns.remove('user_handle')
    df = df[['user_handle'] + columns]
    df['total_time_spent'] = df.sum(axis=1)

    # Sort data frame using user_handle
    df = df.sort_values('user_handle').reset_index(drop=True)

    # Convert time spent into percentage of total time spent
    for col in df.columns:
        if col in ['total_time_spent', 'user_handle']:
            continue
        df[col] = (df[col] * 100) / df['total_time_spent']
        df[col] = df[col].round(decimals=2)

    df.drop('total_time_spent', axis=1, inplace=True)

    # Convert table to matrix
    feature_matrix = df.values

    # Init nearest neighbors object
    nbrs = NearestNeighbors(n_neighbors=3, algorithm='auto').fit(feature_matrix[0:, 1:])
    # Find distance and index of similar users
    distances, indices = nbrs.kneighbors(
        df[df['user_handle'] == user_handle].values[0:, 1:], n_neighbors=3)

    # Map index of matrix to user handle
    matched_user_handles = list(feature_matrix[indices[0]][:, 0].astype(int))

    # List to store json response
    matching_users = []

    # Iterate over similar results
    for distance, matched_user_handle in zip(distances[0], matched_user_handles):

        # Skip same user
        if distance == 0:
            continue

        courses, tags = get_courses_and_tags(matched_user_handle)
        # Add distance and similar result
        matching_users.append({
            'distance': int(distance),
            'user_handle': int(matched_user_handle),
            'courses': courses,
            'tags': tags
        })

    # Set json response
    json_response = {}
    json_response['matching_users'] = matching_users
    courses, tags = get_courses_and_tags(user_handle)

    json_response['queried_user'] = {
        'user_handle': user_handle,
        'tags': tags,
        'courses': courses
    }

    # Return json response
    return json_response


@app.route('/find-similar-users/<user_handle>')
def index(user_handle):
    return jsonify(get_similar_user(int(user_handle))), 200


if __name__ == '__main__':
    get_db_connection()
    import_data_to_db()
    app.run(debug=True)
