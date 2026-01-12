import asyncio
import aiofiles
from typing import List, Optional


async def read_file_async(filepath: str) -> Optional[str]:
    """
    Read a single file asynchronously.
    Returns file content as string, or None if file cannot be read.
    """
    try:
        async with aiofiles.open(filepath, mode='r', encoding='utf-8') as f:
            content = await f.read()
            return content
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None


async def read_files_async(filepaths: List[str]) -> List[Optional[str]]:
    """
    Read multiple files asynchronously.
    Returns list of file contents in the same order as input filepaths.
    If a file cannot be read, None is placed at that position.
    """
    tasks = [read_file_async(filepath) for filepath in filepaths]
    results = await asyncio.gather(*tasks)
    return list(results)


async def main():
    # Example usage
    files = ['file.txt', 'file3.txt','file2.txt', 'nonexistent.txt']
    
    print("Reading files asynchronously...")
    output = await read_files_async(files)
    
    # print("\nResults:")
    # for i, (filename, content) in enumerate(zip(files, output)):
    #     if content is not None:
    #         print(f"{i+1}. {filename}: Read {len(content)} characters")
    #     else:
    #         print(f"{i+1}. {filename}: Failed to read (None)")
    
    return output


# Run the async function
if __name__ == "__main__":
    output = asyncio.run(main())
    print("\nOutput list:", output)