"""Inspect official VLMOD file layout without modifying originals.
Run: python scripts/inspect_official_data.py
Creates reports/official_data_inspection.txt and reports/official_data_inspection.json.
"""
import json, os, re
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'VLMOD(train+test)'
DIRS = {
    'img/train': DATA/'img'/'train',
    'img/test': DATA/'img'/'test',
    'MonoMulti3D/train': DATA/'MonoMulti3D'/'train',
    'MonoMulti3D/test': DATA/'MonoMulti3D'/'test',
}
IMAGE_EXT = {'.jpg','.jpeg','.png','.bmp','.webp','.tif','.tiff'}
SAMPLE_LIMIT = 6
TEXT_LIMIT = 2400
MAX_RECURSIVE_COUNT = 200000


def inspect_directory(path, name):
    result={'directory':str(path),'exists':path.is_dir()}
    if not path.is_dir(): return result
    files=[]; subdirs=0
    for dirpath, dirs, names in os.walk(path):
        subdirs += len(dirs)
        for f in names:
            files.append(Path(dirpath)/f)
            if len(files)>=MAX_RECURSIVE_COUNT:break
        if len(files)>=MAX_RECURSIVE_COUNT:break
    files=sorted(files)
    result.update(file_count=len(files),subdir_count=subdirs, possibly_truncated=len(files)>=MAX_RECURSIVE_COUNT,
                  extensions=dict(Counter(p.suffix.lower() or '<none>' for p in files)),
                  samples=[str(p.relative_to(path)) for p in files[:15]],
                  stems_sample=[p.stem for p in files[:SAMPLE_LIMIT]])
    if name.startswith('MonoMulti3D/'):
        candidates=[p for p in files if p.suffix.lower() in {'.txt','.json','.jsonl','.csv'}]
        chosen=candidates[:SAMPLE_LIMIT]
        # also collect last few, in case file names are repetitive but contents differ
        if len(candidates)>SAMPLE_LIMIT: chosen+=candidates[-2:]
        result['annotation_samples']=[]
        for p in chosen:
            raw=p.read_bytes()[:10000]
            encoding='utf-8-sig'
            try: content=raw.decode(encoding)
            except UnicodeDecodeError:
                encoding='gb18030'
                try:content=raw.decode(encoding)
                except UnicodeDecodeError: content=raw.decode('latin-1');encoding='latin-1'
            result['annotation_samples'].append({'file':str(p.relative_to(path)), 'size_bytes':p.stat().st_size,
                'encoding_guess':encoding,'first_characters':content[:TEXT_LIMIT],
                'nonempty_lines_in_sample':len([l for l in content.splitlines() if l.strip()])})
    return result,files


def main():
    reports=ROOT/'reports';reports.mkdir(exist_ok=True)
    report={'root':str(ROOT),'official_dataset_root':str(DATA),'folders':{},'matching':{},'notes':[
        'Only sample TXT contents are recorded. No source image or annotation is modified.',
        'Matching by filename stem is diagnostic only; official annotation may refer to images by other IDs.',
        'Do not train or submit before verifying the official annotation and submission schemas.'
    ]}
    files_map={}
    for label,path in DIRS.items():
        out=inspect_directory(path,label)
        if isinstance(out,tuple): info,files=out
        else: info,files=out,[]
        report['folders'][label]=info;files_map[label]=files
    for split in ('train','test'):
        imgs={p.stem for p in files_map['img/'+split] if p.suffix.lower() in IMAGE_EXT}
        labs={p.stem for p in files_map['MonoMulti3D/'+split] if p.suffix.lower()=='.txt'}
        matched=imgs&labs
        report['matching'][split]={'image_stems':len(imgs),'txt_stems':len(labs),'exact_stem_matches':len(matched),
            'image_only_examples':sorted(imgs-labs)[:10],'txt_only_examples':sorted(labs-imgs)[:10],
            'matched_examples':sorted(matched)[:10]}
    (reports/'official_data_inspection.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['VLMOD 官方数据结构检查报告','项目目录: '+str(ROOT),'官方数据目录: '+str(DATA),'']
    for label,info in report['folders'].items():
        lines += ['='*72,label,'存在: '+str(info['exists'])]
        if info['exists']:
            lines += ['文件数: '+str(info['file_count']),'子目录数: '+str(info['subdir_count']),
                      '扩展名统计: '+str(info['extensions']),'文件名示例: '+str(info['samples'])]
            for s in info.get('annotation_samples',[]):
                lines += ['-'*30,'TXT样本: '+s['file'],'文件大小: '+str(s['size_bytes'])+' bytes',
                          '检测编码: '+s['encoding_guess'],'前 '+str(TEXT_LIMIT)+' 字符:','```',s['first_characters'],'```']
        lines.append('')
    lines += ['='*72,'图片与TXT按同名文件对应的初步检查']
    for split,info in report['matching'].items():lines += [split+': '+json.dumps(info,ensure_ascii=False)]
    lines += ['','提示：同名不一定等于真实匹配，以标注中的图片ID为准。']
    txt=reports/'official_data_inspection.txt';txt.write_text('\n'.join(lines),encoding='utf-8')
    print('检查完成，输出：',txt)
    for k,v in report['folders'].items():print(f'{k}: exists={v["exists"]}, files={v.get("file_count",0)}')
    for k,v in report['matching'].items():print(f'{k}: same-name images/TXT={v["exact_stem_matches"]}')
    print('把 reports/official_data_inspection.txt 上传到对话，即可分析真实字段并编写转换器。')

if __name__=='__main__':main()
