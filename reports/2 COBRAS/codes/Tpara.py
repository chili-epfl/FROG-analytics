sp = modified_superparagraph
p = paragraph_to_insert
sp1, sp2 = sp.split_at(p.position)
if len(sp1) >= 2 or len(sp2) >= 2:
  remove(sp)
  if len(sp1) >= 2:
    insert(sp1)
  insert(new_sp(text, content=p))
  if len(sp2) >= 2:
    insert(sp2)
else:
  sp.insert(p); sp.is_newline=False