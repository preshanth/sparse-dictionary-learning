"""Download FIRST survey FITS files from NRAO archive"""

from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse
from ftplib import FTP
import argparse
import requests
from tqdm import tqdm


class FIRSTArchiveCrawler:
    """Crawls FIRST archive to find FITS files"""
    
    BASE_URL = "ftp://ftp.cv.nrao.edu/first/"
    
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or self.BASE_URL
    
    def list_ra_directories(self) -> List[str]:
        """List RA hour directories (00 to 23)"""
        return [f"{i:02d}" for i in range(24)]
    
    def list_subdirectories(self, ra_dir: str) -> List[str]:
        """List subdirectories within an RA hour directory"""
        return ['00']  # Assuming only '00' subdir
    
    def list_fits_in_subdir(self, ra_dir: str, subdir: str) -> List[str]:
        """List FITS files in a subdirectory"""
        try:
            ftp = FTP('ftp.cv.nrao.edu')
            ftp.login()
            ftp.cwd(f'/first/{ra_dir}/{subdir}')
            
            files = ftp.nlst()
            ftp.quit()
            
            return [f for f in files if f.endswith('.gz')]
        except Exception as e:
            return []
    
    def get_all_fits_files(self, max_files: Optional[int] = None) -> List[tuple]:
        """Get list of (ra_dir, subdir, filename) tuples
        
        Args:
            max_files: Maximum number of files to return
            
        Returns:
            List of (ra_dir, subdir, filename) tuples
        """
        ra_dirs = self.list_ra_directories()
        
        all_files = []
        for ra_dir in tqdm(ra_dirs, desc="Scanning RA directories"):
            subdirs = self.list_subdirectories(ra_dir)
            
            for subdir in subdirs:
                files = self.list_fits_in_subdir(ra_dir, subdir)
                for f in files:
                    all_files.append((ra_dir, subdir, f))
                    if max_files is not None and len(all_files) >= max_files:
                        return all_files
        
        return all_files


    def download_file(self, url: str, output_path: Path, chunk_size: int = 8192) -> bool:
        """Download a single file with progress bar"""

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if url.startswith('ftp://'):
            try:
                parsed = urlparse(url)
                ftp = FTP(parsed.hostname)
                ftp.login()

                ftp.voidcmd('TYPE I')

                with open(output_path, 'wb') as f:
                    with tqdm(unit='B', unit_scale=True, 
                              desc=output_path.name, leave=False) as pbar:
                        def callback(data):
                            f.write(data)
                            pbar.update(len(data))

                        ftp.retrbinary(f'RETR {parsed.path}', callback)

                ftp.quit()
                return True

            except Exception as e:
                if output_path.exists():
                    output_path.unlink()
                return False

        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(output_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, 
                          desc=output_path.name, leave=False) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            return True

        except requests.RequestException as e:
            if output_path.exists():
                output_path.unlink()
            return False


    def download_first_archive(
        self,
        output_dir: str,
        max_fields: Optional[int] = None,
        skip_existing: bool = True
    ) -> int:
        """Download FIRST FITS files from archive
        
        Args:
            output_dir: Local directory to store files
            max_fields: Maximum number of FITS files to download
            skip_existing: Skip files that already exist locally
            
        Returns:
            Number of files successfully downloaded
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        crawler = FIRSTArchiveCrawler()
        
        print("Scanning FIRST archive...")
        all_files = crawler.get_all_fits_files(max_files=max_fields)
        
        print(f"Found {len(all_files)} FITS files to download")
        
        downloaded = 0
        skipped = 0
        
        for ra_dir, subdir, filename in tqdm(all_files, desc="Downloading"):
            local_path = output_path / ra_dir / subdir / filename
            
            if skip_existing and local_path.exists():
                skipped += 1
                continue
            
            url = f"ftp://ftp.cv.nrao.edu/first/{ra_dir}/{subdir}/{filename}"
            
            if self.download_file(url, local_path):
                downloaded += 1
        
        print(f"\nDownload complete:")
        print(f"  Downloaded: {downloaded}")
        print(f"  Skipped (existing): {skipped}")
        print(f"  Total files: {downloaded + skipped}")
        
        return downloaded

if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Download FIRST survey FITS files from NRAO archive")
    parser.add_argument("--output-dir", type=str, help="Directory to save downloaded FITS files")
    parser.add_argument("--max-fields", type=int, default=None, help="Maximum number of FITS files to download")
    parser.add_argument("--skip-existing", action="store_true", help="Skip files that already exist locally")

    args = parser.parse_args()
    print(f"Output directory: {args.output_dir}")
    print(f"Max fields to download: {args.max_fields}")
    print(f"Skip existing files: {args.skip_existing}")
    first_archive_crawler = FIRSTArchiveCrawler()
    first_archive_crawler.download_first_archive(
        output_dir=args.output_dir,
        max_fields=args.max_fields,
        skip_existing=args.skip_existing
    )