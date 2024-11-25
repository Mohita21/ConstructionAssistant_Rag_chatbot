#!/bin/bash

# Define colors for better readability in terminal output
# These make important messages stand out to users
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # NC = No Color, resets text color back to default

echo -e "${BLUE}Starting Materials RAG Application...${NC}"

# First, let's make sure Python is installed
# Python is essential as it runs our backend server
if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed. Please install Python3 to continue."
    exit 1
fi

# This function handles Node.js installation across different operating systems
# Node.js is needed to run our React frontend
install_node() {
    echo -e "${BLUE}We couldn't find Node.js on your system. Let's install it...${NC}"
    
    # Check what operating system we're running on and install accordingly
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # For Linux systems, we'll try different package managers
        # Most Linux distributions use one of these
        if command -v apt &> /dev/null; then
            echo "Found apt package manager - this looks like Ubuntu/Debian..."
            curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
            sudo apt-get install -y nodejs
        elif command -v dnf &> /dev/null; then
            echo "Found dnf package manager - this looks like Fedora..."
            sudo dnf install -y nodejs
        elif command -v yum &> /dev/null; then
            echo "Found yum package manager - this looks like CentOS/RHEL..."
            curl -fsSL https://rpm.nodesource.com/setup_lts.x | sudo bash -
            sudo yum install -y nodejs
        else
            echo -e "${RED}Hmm... We couldn't recognize your package manager. You'll need to install Node.js manually.${NC}"
            exit 1
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # For macOS, we use Homebrew - the most popular package manager for Mac
        if command -v brew &> /dev/null; then
            echo "Great! You have Homebrew installed. Let's use it to install Node.js..."
            brew install node
        else
            echo "Installing Homebrew first (it's like apt-get for Mac)..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            brew install node
        fi
    else
        echo -e "${RED}Sorry, we don't recognize your operating system. Please install Node.js manually.${NC}"
        exit 1
    fi
}

# Check if Node.js is installed, if not, let's install it
if ! command -v node &> /dev/null; then
    install_node
    # Double-check that the installation worked
    if ! command -v node &> /dev/null; then
        echo -e "${RED}Something went wrong with the Node.js installation. Please try installing it manually.${NC}"
        exit 1
    fi
    echo -e "${GREEN}Great! Node.js is now installed and ready to go!${NC}"
fi

# This function checks if a specific port is already being used
# We need ports 3000 and 8000 to be free for our application
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null ; then
        return 0
    else
        return 1
    fi
}

# Make sure our required ports are available
# Port 3000 is for the React frontend
if check_port 3000; then
    echo "Oops! Port 3000 is already being used. Please close whatever is using it and try again."
    exit 1
fi

# Port 8000 is for the FastAPI backend
if check_port 8000; then
    echo "Oops! Port 8000 is already being used. Please close whatever is using it and try again."
    exit 1
fi

# Let's start the backend server first
echo -e "${GREEN}Starting up the FastAPI Backend Server...${NC}"
cd "$(dirname "$0")"
# Install Python dependencies quietly in the background
python3 -m pip install -r requirements.txt > /dev/null 2>&1 &
python3 api.py &
BACKEND_PID=$!

# Wait for the backend to be fully ready
echo "Waiting for the backend server to warm up..."
until [ "$(curl -s http://localhost:8000/health)" = '{"status":"healthy"}' ]; do
    printf '.'
    sleep 1
done
echo -e "\nGreat! Backend is up and running!"

# Now let's start the frontend
echo -e "${GREEN}Starting up the React Frontend Server...${NC}"
cd materials-rag-chat
# Install necessary npm packages quietly in the background
npm install react-markdown > /dev/null 2>&1
npm install > /dev/null 2>&1
npm start &
FRONTEND_PID=$!

# This function handles graceful shutdown when the user presses Ctrl+C
cleanup() {
    echo -e "${BLUE}Shutting everything down gracefully...${NC}"
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

# Set up our cleanup function to run when the script is interrupted
trap cleanup SIGINT SIGTERM

# Let the user know everything is ready to go!
echo -e "${GREEN}Success! Everything is up and running!${NC}"
echo "You can access the frontend at: http://localhost:3000"
echo "The backend is running at: http://localhost:8000"
echo "To stop everything, just press Ctrl+C"

# Keep the script running and waiting for both processes
wait $BACKEND_PID $FRONTEND_PID