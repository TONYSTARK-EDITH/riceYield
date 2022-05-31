# Generated by Django 4.0.4 on 2022-05-16 19:04

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='DTR',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('predicted', models.FloatField()),
            ],
            options={
                'verbose_name_plural': 'DTR',
            },
        ),
        migrations.CreateModel(
            name='Lasso',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('predicted', models.FloatField()),
            ],
            options={
                'verbose_name_plural': 'lasso',
            },
        ),
        migrations.CreateModel(
            name='Mlmodel',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('nitrogen', models.FloatField()),
                ('phosphorus', models.FloatField()),
                ('pottasium', models.FloatField()),
                ('rainfall', models.FloatField()),
                ('rice_yield', models.FloatField()),
            ],
            options={
                'verbose_name_plural': 'MlModel',
            },
        ),
        migrations.CreateModel(
            name='Mlr',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('predicted', models.FloatField()),
            ],
            options={
                'verbose_name_plural': 'mlr',
            },
        ),
        migrations.CreateModel(
            name='RealValue',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('realVal', models.FloatField()),
            ],
            options={
                'verbose_name_plural': 'RealValue',
            },
        ),
        migrations.CreateModel(
            name='Report',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.TextField()),
                ('which', models.TextField(unique=True)),
                ('user', models.TextField()),
                ('n', models.FloatField()),
                ('p', models.FloatField()),
                ('k', models.FloatField()),
                ('rain', models.FloatField()),
                ('area', models.FloatField()),
                ('pred', models.FloatField()),
                ('month', models.FloatField()),
                ('moist', models.FloatField()),
                ('temp', models.FloatField()),
                ('crop', models.TextField()),
            ],
            options={
                'verbose_name_plural': 'Report',
            },
        ),
        migrations.CreateModel(
            name='RF',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('predicted', models.FloatField()),
            ],
            options={
                'verbose_name_plural': 'RF',
            },
        ),
        migrations.CreateModel(
            name='RIDGE',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('predicted', models.FloatField()),
            ],
            options={
                'verbose_name_plural': 'ridge',
            },
        ),
        migrations.CreateModel(
            name='Svr',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('predicted', models.FloatField()),
            ],
            options={
                'verbose_name_plural': 'SVR',
            },
        ),
    ]
