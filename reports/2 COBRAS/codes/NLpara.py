sp = modified_superparagraph
p = paragraph_to_insert
if (len(sp.paras) == 1 and
  sp.ends_by_newline):
    sp.insert(p); sp.is_newline=True
elif ((p.at_start_of(sp) and
  sp.starts_by_newline) or
  (p.at_end_of(sp) and
  sp.ends_by_newline)):
  insert(new_sp(newline, len=2))
elif p.at_middle_of(sp):
  sp1, sp2 = sp.split_at(p.position)
  if sp1.not_empty and sp2.not_empty:
    remove(sp)
    insert(sp1)
    insert(new_sp(newline, len=2))
    insert(sp2)
  else:
    insert(new_sp(newline, len=2))
else:
  sp.insert(p)