"""
Setup script for MoodTune
Initializes directories and checks dependencies
"""

import os
import sys

def create_directories():
    """Create required directories"""
    directories = [
        'data',
        'logs',
        'static/uploads',
        'config',
        'cv',
        'rl',
        'utils',
        'templates',
        'static/css',
        'static/js'
    ]
    
    print("Creating directories...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created: {directory}")

def create_init_files():
    """Create __init__.py files for packages"""
    packages = ['config', 'cv', 'rl', 'utils']
    
    print("\nCreating __init__.py files...")
    for package in packages:
        init_file = os.path.join(package, '__init__.py')
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write(f'"""\n{package.capitalize()} package for MoodTune\n"""\n')
            print(f"✓ Created: {init_file}")

def check_env_file():
    """Check if .env file exists"""
    print("\nChecking environment configuration...")
    
    if not os.path.exists('.env'):
        print("⚠️  .env file not found. Creating template...")
        
        env_template = """# Spotify API Credentials
SPOTIPY_CLIENT_ID=your_spotify_client_id_here
SPOTIPY_CLIENT_SECRET=your_spotify_client_secret_here

# TMDB API Key
TMDB_API_KEY=your_tmdb_api_key_here

# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=moodtune_secret_key_2024
"""
        
        with open('.env', 'w') as f:
            f.write(env_template)
        
        print("✓ Created .env template")
        print("⚠️  Please update .env with your API keys!")
    else:
        print("✓ .env file exists")

def check_dependencies():
    """Check if required packages are installed"""
    print("\nChecking dependencies...")
    
    required_packages = [
        'flask',
        'opencv-python',
        'fer',
        'mtcnn',
        'numpy',
        'pillow',
        'tensorflow',
        'spotipy',
        'requests',
        'python-dotenv',
        'flask-cors'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All dependencies installed")
        return True

def main():
    """Main setup function"""
    print("=" * 60)
    print("🎵 MoodTune Setup Script 🎵")
    print("=" * 60)
    print()
    
    # Create directories
    create_directories()
    
    # Create __init__.py files
    create_init_files()
    
    # Check .env file
    check_env_file()
    
    # Check dependencies
    dependencies_ok = check_dependencies()
    
    print("\n" + "=" * 60)
    
    if dependencies_ok:
        print("✓ Setup complete! Ready to run MoodTune.")
        print("\nNext steps:")
        print("1. Update .env with your API keys")
        print("2. Run: python app.py")
        print("3. Open: http://localhost:5000")
    else:
        print("⚠️  Setup incomplete. Please install missing dependencies.")
        print("Run: pip install -r requirements.txt")
    
    print("=" * 60)

if __name__ == '__main__':
    main()