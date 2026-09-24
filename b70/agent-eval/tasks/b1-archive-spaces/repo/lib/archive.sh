# Helpers for bin/archive-logs. Source this file; do not run it.

# Print the *.log files directly inside a directory, one per line.
list_logs() {
    ls $1/*.log 2>/dev/null
}

# Gzip one file into a destination directory as <basename>.gz and
# print a line saying so.
archive_one() {
    local src=$1 dest=$2
    local name=$(basename $src)
    gzip -c $src > $dest/$name.gz
    echo "archived $name"
}
