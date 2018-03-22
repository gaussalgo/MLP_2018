import cdsw, socket, json, pickle
import time

# # Master start
started = time.time()

WORKERS = 3
FOLDER = 'data/unique_1k_images/'

# Launch two CDSW workers. These are engines that will run in 
# the same project, execute a given code or script, and exit.
print('Launch workers')
workers = cdsw.launch_workers(n=WORKERS, cpu=1, memory=2, script="src/is_worker.py")


# Listen on TCP port 6000
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("0.0.0.0", 6000))
s.listen(5)

# # Accept as connections from workers, send them config or receive results
print('Workers management')
worker_files = {}
worker_id = 0
while True:
    conn, addr = s.accept()
    data_str = conn.recv(2048)
    assert data_str
    data = json.loads(data_str)
    
    if data['type'] == 'started':
        print('Worker init: %s' % data)

        cfg = {
            'folder_path': FOLDER,
            'worker_id': worker_id,
            'workers': WORKERS,
        }
        conn.send(json.dumps(cfg))
        worker_id += 1
    elif data['type'] == 'finished':
        # expect {'worker_id': int, 'filename': str}
        print('Worker collect: %s' % data)
        worker_files[data['worker_id']] = data['filename']
    else:
        assert 0
        
    conn.close()
    
    if len(worker_files) == WORKERS:
        break

s.close()

# # Concatenate worker results and save
embedding = []
file_paths = []
for worker_id, workerfile in sorted(worker_files.items()):
    with open(workerfile, 'rb') as f:
        worker_data = pickle.load(f)
        embedding.extend(worker_data['embedding'])
        file_paths.extend(worker_data['file_paths'])
    os.unlink(workerfile)

image_vectors_file = 'data/image_vectors.pkl'
with open(image_vectors_file, 'wb') as f:
    data = {
        'embedding': embedding,
        'file_paths': file_paths,
    }
    pickle.dump(data, f)

# # Master DONE
time_diff = int(time.time() - started)
print('Master finished, took %d min %d sec' % (time_diff / 60, time_diff % 60))
