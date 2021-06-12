for i in range(50):
	filename = f'rules_{i}.txt'
	print(filename)
	rules = []
	for line in open(filename):
		a = line.split('\t')
		if len(a) != 2:
			continue
		rule, prec = a
		rules.append((rule, prec))

	rules = sorted(rules, key=lambda x : float(x[-1]))[:10000]

	file = open(filename, 'w')
	for rule, prec in rules:
		file.write(f'{rule}\t{prec}')
	file.close()