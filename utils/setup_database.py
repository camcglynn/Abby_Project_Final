import os
import logging
from models.user import db, User

logger = logging.getLogger(__name__)

def setup_database():
    """
    Set up the database tables and create the admin user if it doesn't exist
    """
    try:
        # Create database tables
        db.create_all()
        logger.info("Database tables created")
        
        # Check if admin user exists
        admin_username = os.environ.get('ADMIN_USERNAME', 'admin')
        admin_exists = User.query.filter_by(username=admin_username).first()
        
        if not admin_exists:
            # Create admin user
            admin_password = os.environ.get('ADMIN_PASSWORD', 'adminpassword')
            admin_user = User(
                username=admin_username,
                password=admin_password,
                is_admin=True
            )
            db.session.add(admin_user)
            db.session.commit()
            logger.info(f"Admin user '{admin_username}' created successfully")
        else:
            logger.info(f"Admin user '{admin_username}' already exists")
        
        return True
    except Exception as e:
        logger.error(f"Error setting up database: {str(e)}")
        return False 